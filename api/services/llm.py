from django.conf import settings
from openai import AsyncOpenAI
from typing import List, Dict, Any, Generator, Optional, Callable, Union, AsyncGenerator
import logging
import json
import aiohttp
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Default model can be set in settings, e.g., 'anthropic/claude-3.7-sonnet:thinking'
# Or keep a default like this if OPENROUTER_API_MODEL is not set
default_model = getattr(settings, 'OPENROUTER_API_MODEL', 'google/gemini-2.5-pro')

async def stream_text(
    prompt: str,
    model: str = default_model,
    max_tokens: int = 4096, # Adjusted default, OpenRouter/OpenAI often have different limits/recommendations
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
    site_url: Optional[str] = getattr(settings, 'OPENROUTER_SITE_URL', None), # Optional: For leaderboard ranking
    site_title: Optional[str] = getattr(settings, 'OPENROUTER_SITE_TITLE', None), # Optional: For leaderboard ranking
    response_schema: Optional[Dict[str, Any]] = None, # New: JSON schema for structured output
    schema_name: Optional[str] = None, # New: Optional name for the schema (used in OpenRouter's format)
    schema_strict: bool = True, # New: Enforce strict schema adherence (recommended by OpenRouter)
    include_reasoning: bool = False, # New: Request reasoning tokens
    should_use_anakin: bool = False, # New: Whether to use Anakin API instead of OpenRouter
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text responses from OpenRouter API asynchronously using the OpenAI SDK compatibility.
    Supports optional structured output via JSON Schema and reasoning token inclusion.

    Args:
        prompt: The user prompt to send to the model.
        model: The model identifier on OpenRouter (e.g., 'anthropic/claude-3.7-sonnet:thinking', 'anthropic/claude-3.5-sonnet').
        max_tokens: Maximum number of tokens in the response.
        system_prompt: Optional system prompt.
        messages: Optional list of message objects (overrides prompt if provided).
        callback: Optional async callback function to process streaming events.
        site_url: Optional site URL for OpenRouter leaderboards.
        site_title: Optional site title for OpenRouter leaderboards.
        response_schema: Optional dictionary representing the JSON Schema for the desired output format.
        schema_name: Optional name for the schema (required by OpenRouter's structure if response_schema is used).
        schema_strict: If using response_schema, whether to enforce strict adherence. Defaults to True.
        include_reasoning: Whether to request reasoning tokens (supported by specific models).

    Yields:
        Dictionary containing event information for each streaming event (OpenAI format, potentially with a 'reasoning' field).
    """
    logger.info(f"Starting async stream_text with OpenRouter model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")

    # --- Ensure event loop has a default ThreadPoolExecutor (OpenAI SDK calls asyncio.to_thread) ---
    try:
        loop = asyncio.get_running_loop()
        if getattr(loop, "_default_executor", None) is None:
            import concurrent.futures
            loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=4))
            logger.debug("Set new default ThreadPoolExecutor for event loop")
    except RuntimeError:
        # No running loop (shouldn't happen in async context), ignore
        pass

    # Also check if the executor has been shut down and needs replacement
    try:
        loop = asyncio.get_running_loop()
        executor = getattr(loop, "_default_executor", None)
        if executor and hasattr(executor, "_shutdown") and executor._shutdown:
            import concurrent.futures
            new_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            loop.set_default_executor(new_executor)
            logger.debug("Replaced shut down ThreadPoolExecutor with new one")
    except Exception as e:
        logger.debug(f"Could not check/replace executor: {e}")

    if should_use_anakin:
        print("Using Anakin API")
        async for chunk in stream_text_anakin(prompt, model, max_tokens, system_prompt, messages, callback):
            yield chunk
        return

    try:
        # Check if API key is configured
        if not hasattr(settings, 'OPENROUTER_API_KEY') or not settings.OPENROUTER_API_KEY:
            logger.error("OPENROUTER_API_KEY is not configured in settings")
            raise ValueError("OPENROUTER_API_KEY is not configured")

        logger.debug("Initializing AsyncOpenAI client for OpenRouter")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
        )

        # Configure messages
        if messages is None:
            logger.debug("Using single prompt message")
            messages_config = [{"role": "user", "content": prompt}]
        else:
            logger.debug(f"Using provided messages array with {len(messages)} messages")
            messages_config = messages

        # Add system prompt if provided and messages_config doesn't already have one
        if system_prompt and not any(msg['role'] == 'system' for msg in messages_config):
             logger.debug("Prepending system prompt")
             messages_config.insert(0, {"role": "system", "content": system_prompt})
        elif system_prompt:
             logger.warning("System prompt provided but messages_config already contains a system message. Ignoring provided system_prompt argument.")


        # Prepare standard stream parameters
        stream_params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages_config,
            "stream": True,
        }

        # --- Prepare extra_body for non-standard params ---
        extra_body: Dict[str, Any] = {}

        # Add structured output configuration if schema is provided
        # Note: The standard 'response_format' might need to go in extra_body too if not supported directly by SDK version
        # Let's try keeping it direct first, as it's more standard OpenAI API now.
        if response_schema:
            schema_name_to_use = schema_name or "custom_schema"
            if not schema_name:
                 logger.warning("No schema_name provided for structured output, using default: 'custom_schema'")
            structured_output_config = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name_to_use,
                    "strict": schema_strict,
                    "schema": response_schema
                }
            }
            # Check if 'response_format' is a known param, if not, move to extra_body
            # For now, assume it's standard and keep in stream_params
            stream_params["response_format"] = structured_output_config
            logger.info(f"Using structured output with schema name: {schema_name_to_use}, strict: {schema_strict}")


        # --- Add reasoning parameter to extra_body ---
        if include_reasoning:
            # Pass reasoning={} in extra_body
            extra_body["reasoning"] = {}
            logger.info("Requesting reasoning tokens via extra_body={'reasoning': {}}.")


        # Add optional headers for OpenRouter ranking
        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_title:
            extra_headers["X-Title"] = site_title
        # Add extra_headers to stream_params if they exist
        if extra_headers:
             stream_params["extra_headers"] = extra_headers


        logger.info("Starting async stream with OpenRouter API")
        # Log parameters, including extra_body if present
        loggable_params = {k: v for k, v in stream_params.items() if k not in ['messages', 'response_format']}
        if "response_format" in stream_params:
            loggable_params["response_format_type"] = stream_params["response_format"].get("type")
            loggable_params["schema_name"] = stream_params["response_format"].get("json_schema", {}).get("name")
        if extra_body:
             loggable_params["extra_body"] = extra_body # Log extra_body content
        logger.debug(f"Stream parameters (excluding messages/schema): {loggable_params}")

        try:
            # Pass extra_body to the create call
            stream = await client.chat.completions.create(**stream_params, extra_body=extra_body if extra_body else None)
            logger.info("Stream connection established")
            async for chunk in stream:
                # Log the event type (chunk usually has choices)
                logger.debug(f"Received chunk: {chunk.id}")

                # --- ADDED: Check for reasoning in the chunk based on docs example ---
                reasoning_content = None
                if chunk.choices and hasattr(chunk.choices[0].delta, 'reasoning'):
                    reasoning_content = chunk.choices[0].delta.reasoning
                    if reasoning_content:
                         logger.debug(f"Chunk contains reasoning delta: {reasoning_content}")
                # --- END ADDED ---

                # Process the event with callback
                if callback:
                    logger.debug("Calling user-provided callback")
                    await callback(chunk) # Callback needs to handle chunk structure

                # Yield the chunk (OpenAI object) to the caller
                yield chunk

            logger.info("Stream completed successfully")

        except Exception as stream_error:
            # General error handling
            logger.error(f"Error during async streaming with OpenRouter: {str(stream_error)}", exc_info=True)
            # Check if the error message indicates lack of support for reasoning *parameter* specifically
            if extra_body.get("reasoning") is not None and ("reasoning" in str(stream_error).lower() or "support" in str(stream_error).lower()):
                 logger.warning(f"Model '{model}' might not support the 'reasoning' parameter via extra_body, or the parameter structure is incorrect.")
            # Check for structured output errors
            if "response_format" in stream_params and "support" in str(stream_error).lower():
                 logger.warning(f"Model '{model}' might not support structured outputs ('response_format'), or the schema might be invalid.")
            raise

    except Exception as e:
        logger.error(f"Error in async stream_text (OpenRouter): {str(e)}", exc_info=True)
        # Re-raise the exception after logging
        raise


async def stream_sambanova(
    prompt: str,
    model: str = getattr(settings, 'SAMBANOVA_MODEL', "Llama-4-Scout-17B-16E-Instruct"), # Default from quickstart if not in settings
    max_tokens: int = 2048, # SambaNova might have different defaults/limits
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text responses from SambaNova API asynchronously using the OpenAI SDK compatibility.

    Args:
        prompt: The user prompt to send to the model.
        model: The model identifier for SambaNova (e.g., 'Meta-Llama-3.1-405B-Instruct').
        max_tokens: Maximum number of tokens in the response.
        system_prompt: Optional system prompt.
        messages: Optional list of message objects (overrides prompt if provided).
        callback: Optional async callback function to process streaming events.

    Yields:
        Dictionary containing event information for each streaming event (OpenAI format).
    """
    logger.info(f"Starting async stream_sambanova with SambaNova model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")

    try:
        # Check if API key is configured
        if not hasattr(settings, 'SAMBANOVA_API_KEY') or not settings.SAMBANOVA_API_KEY:
            logger.error("SAMBANOVA_API_KEY is not configured in settings")
            raise ValueError("SAMBANOVA_API_KEY is not configured")

        sambanova_api_key = settings.SAMBANOVA_API_KEY
        sambanova_base_url = "https://api.sambanova.ai/v1" # As per SambaNova documentation

        logger.debug(f"Initializing AsyncOpenAI client for SambaNova: {sambanova_base_url}")
        client = AsyncOpenAI(
            base_url=sambanova_base_url,
            api_key=sambanova_api_key,
        )

        # Configure messages
        if messages is None:
            logger.debug("Using single prompt message for SambaNova")
            messages_config = [{"role": "user", "content": prompt}]
        else:
            logger.debug(f"Using provided messages array with {len(messages)} messages for SambaNova")
            messages_config = messages

        # Add system prompt if provided and messages_config doesn't already have one
        if system_prompt and not any(msg['role'] == 'system' for msg in messages_config):
             logger.debug("Prepending system prompt for SambaNova")
             messages_config.insert(0, {"role": "system", "content": system_prompt})
        elif system_prompt:
             logger.warning("System prompt provided for SambaNova but messages_config already contains a system message. Ignoring provided system_prompt argument.")

        # Prepare standard stream parameters
        stream_params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages_config,
            "stream": True,
        }
        # SambaNova's example uses "stop": ["<|eot_id|>"] for Llama 3.1 405B.
        # This might be model-specific. For now, let's not add it globally unless specified.
        # If a specific model requires it, it should be passed in `model_specific_params` or similar.

        logger.info("Starting async stream with SambaNova API")
        loggable_params = {k: v for k, v in stream_params.items() if k != 'messages'}
        logger.debug(f"SambaNova Stream parameters (excluding messages): {loggable_params}")

        try:
            stream = await client.chat.completions.create(**stream_params)
            logger.info("SambaNova stream connection established")
            async for chunk in stream:
                logger.debug(f"SambaNova received chunk: {chunk.id}")

                if callback:
                    logger.debug("Calling user-provided callback for SambaNova chunk")
                    await callback(chunk)

                yield chunk
            logger.info("SambaNova stream completed successfully")

        except Exception as stream_error:
            logger.error(f"Error during async streaming with SambaNova: {str(stream_error)}", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Error in async stream_sambanova: {str(e)}", exc_info=True)
        raise


async def stream_text_anakin(
    prompt: str,
    model: str = None,  # Not used by Anakin but kept for compatibility
    max_tokens: int = 4096,  # Not used by Anakin but kept for compatibility  
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
    thread_id: Optional[str] = None,  # Anakin-specific parameter
    app_id: Optional[str] = None,  # Anakin-specific parameter
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text responses from Anakin API asynchronously, returning OpenAI-compatible format.
    
    Args:
        prompt: The user prompt to send to the model.
        model: Model identifier (not used by Anakin but kept for compatibility).
        max_tokens: Maximum tokens (not used by Anakin but kept for compatibility).
        system_prompt: Optional system prompt (will be prepended to content).
        messages: Optional list of message objects (will be converted to single content).
        callback: Optional async callback function to process streaming events.
        thread_id: Optional Anakin thread ID to continue existing conversation.
        app_id: Anakin app/chatbot ID (defaults to settings.ANAKIN_APP_ID).
        
    Yields:
        Dictionary containing event information in OpenAI format for compatibility.
    """
    logger.info(f"Starting async stream_text_anakin with app_id: {app_id}")
    logger.debug(f"Prompt length: {len(prompt)} characters")

    # --- Ensure event loop has a working ThreadPoolExecutor ---
    try:
        loop = asyncio.get_running_loop()
        executor = getattr(loop, "_default_executor", None)
        
        # Create new executor if none exists or if it's been shut down
        if executor is None or (hasattr(executor, "_shutdown") and executor._shutdown):
            import concurrent.futures
            new_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            loop.set_default_executor(new_executor)
            logger.debug("Set/replaced ThreadPoolExecutor for Anakin stream")
    except Exception as e:
        logger.debug(f"Could not check/set executor for Anakin: {e}")

    try:
        # Check if API key is configured
        if not hasattr(settings, 'ANAKIN_API_KEY') or not settings.ANAKIN_API_KEY:
            logger.error("ANAKIN_API_KEY is not configured in settings")
            raise ValueError("ANAKIN_API_KEY is not configured")

        # Get app_id from settings if not provided
        if not app_id:
            app_id = getattr(settings, 'ANAKIN_APP_ID', None)
            if not app_id:
                logger.error("ANAKIN_APP_ID is not configured in settings and not provided")
                raise ValueError("ANAKIN_APP_ID is not configured")

        anakin_api_key = settings.ANAKIN_API_KEY
        anakin_base_url = "https://api.anakin.ai"
        api_version = getattr(settings, 'ANAKIN_API_VERSION', '2024-05-06')

        # Prepare content from prompt, system_prompt, and messages
        content = ""
        if system_prompt:
            content += f"System: {system_prompt}\n\n"
            
        if messages:
            # Convert messages to single content string
            for msg in messages:
                role = msg.get('role', 'user')
                msg_content = msg.get('content', '')
                if role == 'system' and not system_prompt:  # Only add if no system_prompt was already added
                    content += f"System: {msg_content}\n\n"
                elif role == 'user':
                    content += f"User: {msg_content}\n\n"
                elif role == 'assistant':
                    content += f"Assistant: {msg_content}\n\n"
        else:
            content += prompt

        # Prepare payload
        payload = {
            "content": content.strip(),
            "stream": True
        }
        if thread_id:
            payload["threadId"] = thread_id

        headers = {
            'Authorization': f'Bearer {anakin_api_key}',
            'X-Anakin-Api-Version': api_version,
            'Content-Type': 'application/json'
        }

        logger.info("Starting async stream with Anakin API")
        logger.debug(f"Anakin request payload (content length: {len(payload['content'])})")

        # Create a mock OpenAI-style chunk structure
        def create_openai_chunk(content_delta: str = "", finish_reason: Optional[str] = None, chunk_id: Optional[str] = None):
            """Create OpenAI-compatible chunk structure"""
            if not chunk_id:
                chunk_id = f"anakin-{int(datetime.now().timestamp() * 1000)}"
                
            chunk = type('Chunk', (), {
                'id': chunk_id,
                'object': 'chat.completion.chunk',
                'created': int(datetime.now().timestamp()),
                'model': model or 'anakin-chatbot',
                'choices': [
                    type('Choice', (), {
                        'index': 0,
                        'delta': type('Delta', (), {
                            'content': content_delta,
                            'role': 'assistant' if content_delta else None
                        })(),
                        'finish_reason': finish_reason
                    })()
                ]
            })()
            return chunk

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{anakin_base_url}/v1/chatbots/{app_id}/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Anakin API error {response.status}: {error_text}")
                    raise Exception(f"Anakin API error {response.status}: {error_text}")
                
                logger.info("Anakin stream connection established")
                
                # Handle server-sent events
                accumulated_content = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if line.startswith('data: '):
                        data_content = line[6:]  # Remove 'data: ' prefix
                        
                        if data_content == '[DONE]':
                            # Stream finished
                            final_chunk = create_openai_chunk(finish_reason='stop')
                            if callback:
                                await callback(final_chunk)
                            yield final_chunk
                            break
                            
                        try:
                            # Try to parse as JSON
                            event_data = json.loads(data_content)
                            
                            # Extract content delta
                            if isinstance(event_data, dict):
                                if 'content' in event_data:
                                    # Full content response
                                    new_content = event_data['content']
                                    content_delta = new_content[len(accumulated_content):]
                                    accumulated_content = new_content
                                elif 'delta' in event_data:
                                    # Delta response
                                    content_delta = event_data['delta']
                                    accumulated_content += content_delta
                                else:
                                    # Other event types, send as empty delta
                                    content_delta = ""
                                    
                                # Create OpenAI-compatible chunk
                                chunk = create_openai_chunk(content_delta)
                                
                                if callback:
                                    await callback(chunk)
                                yield chunk
                                
                        except json.JSONDecodeError:
                            # Not JSON, might be plain text delta
                            if data_content:
                                chunk = create_openai_chunk(data_content)
                                if callback:
                                    await callback(chunk)
                                yield chunk
                    
                    elif line.startswith('event: ') or line == '':
                        # SSE event type or empty line, ignore
                        continue
                        
                logger.info("Anakin stream completed successfully")

    except Exception as e:
        logger.error(f"Error in async stream_text_anakin: {str(e)}", exc_info=True)
        raise

