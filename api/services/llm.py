import logging
import json
import aiohttp
import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Any, Generator, Optional, Callable, Union, AsyncGenerator
from datetime import datetime
from sentence_transformers import SentenceTransformer
import instructor
from functools import lru_cache

load_dotenv()

logger = logging.getLogger(__name__)

# Default model can be set in settings, e.g., 'anthropic/claude-3.7-sonnet:thinking'
# Or keep a default like this if OPENROUTER_API_MODEL is not set
default_model = os.getenv('OPENROUTER_API_MODEL', 'google/gemini-2.5-pro')

embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Using a small, fast model suitable for real-time simulation
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

@lru_cache(maxsize=128)
def get_embedding(text, use_local=False):
    if use_local:
        client = OpenAI(
            base_url=os.getenv('OLLAMA_BASE_URL', "http://localhost:11434/v1"),
            api_key='ollama',
        )
        model = "mxbai-embed-large"
        response = client.embeddings.create(model=model, input=[text])
        return response.data[0].embedding
    else:
        # Default to a local sentence-transformer model. It's fast, reliable, 
        # and avoids issues with remote embedding providers that have been failing.
        model = get_embedding_model()
        embedding = model.encode(text)
        return embedding


async def stream_text(
    prompt: str,
    model: str = default_model,
    max_tokens: int = 4096, # Adjusted default, OpenRouter/OpenAI often have different limits/recommendations
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
    site_url: Optional[str] = os.getenv('OPENROUTER_SITE_URL', None), # Optional: For leaderboard ranking
    site_title: Optional[str] = os.getenv('OPENROUTER_SITE_TITLE', None), # Optional: For leaderboard ranking
    response_schema: Optional[Dict[str, Any]] = None, # New: JSON schema for structured output
    schema_name: Optional[str] = None, # New: Optional name for the schema (used in OpenRouter's format)
    schema_strict: bool = True, # New: Enforce strict schema adherence (recommended by OpenRouter)
    include_reasoning: bool = False, # New: Request reasoning tokens
    should_use_anakin: bool = False, # New: Whether to use Anakin API instead of OpenRouter
    should_use_ollama: bool = False,
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

    if should_use_ollama:
        logger.info("Routing to Ollama stream.")
        # If the model is the default OpenRouter one, switch to the default Ollama one. Otherwise, use what's passed.
        ollama_model = model if model != default_model else os.getenv('OLLAMA_MODEL', 'llama3')
        async for chunk in stream_ollama(
            prompt,
            model=ollama_model,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            messages=messages,
            callback=callback,
            response_schema=response_schema
        ):
             yield chunk
        return

    if should_use_anakin:
        print("Using Anakin API")
        async for chunk in stream_text_anakin(prompt, model, max_tokens, system_prompt, messages, callback):
            yield chunk
        return

    try:
        # Check if API key is configured
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.error("OPENROUTER_API_KEY is not configured in settings")
            raise ValueError("OPENROUTER_API_KEY is not configured")

        logger.debug("Initializing AsyncOpenAI client for OpenRouter")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
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
    model: str = os.getenv('SAMBANOVA_MODEL', "Llama-4-Scout-17B-16E-Instruct"), # Default from quickstart if not in settings
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
        sambanova_api_key = os.getenv('SAMBANOVA_API_KEY')
        if not sambanova_api_key:
            logger.error("SAMBANOVA_API_KEY is not configured in settings")
            raise ValueError("SAMBANOVA_API_KEY is not configured")

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


async def stream_ollama(
    prompt: str,
    model: str = os.getenv('OLLAMA_MODEL', "llama3"),
    max_tokens: int = 4096,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable] = None,
    response_schema: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text responses from a local Ollama instance asynchronously.
    Supports basic JSON mode if a response_schema is provided.
    
    Args:
        prompt: The user prompt to send to the model.
        model: The model identifier for Ollama (e.g., 'llama3', 'codellama').
        max_tokens: Maximum number of tokens in the response.
        system_prompt: Optional system prompt.
        messages: Optional list of message objects (overrides prompt if provided).
        callback: Optional async callback function to process streaming events.
        response_schema: If provided, enables JSON mode for the response.
    
    Yields:
        Dictionary containing event information for each streaming event (OpenAI format).
    """
    logger.info(f"Starting async stream_ollama with model: {model}")
    
    try:
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', "http://localhost:11434/v1")
        
        logger.debug(f"Initializing AsyncOpenAI client for Ollama: {ollama_base_url}")
        client = AsyncOpenAI(
            base_url=ollama_base_url,
            api_key='ollama', # Required by the library but not used by Ollama
        )

        # Configure messages
        if messages is None:
            logger.debug("Using single prompt message for Ollama")
            messages_config = [{"role": "user", "content": prompt}]
        else:
            logger.debug(f"Using provided messages array with {len(messages)} messages for Ollama")
            messages_config = messages

        # Add system prompt
        if system_prompt and not any(msg['role'] == 'system' for msg in messages_config):
             logger.debug("Prepending system prompt for Ollama")
             messages_config.insert(0, {"role": "system", "content": system_prompt})

        # Prepare stream parameters
        stream_params: Dict[str, Any] = {
            "model": model,
            "messages": messages_config,
            "stream": True,
            "max_tokens": max_tokens,
        }

        # Enable JSON mode if schema is provided
        if response_schema:
            stream_params["response_format"] = {"type": "json_object"}
            logger.info("Ollama JSON mode enabled.")

        logger.info("Starting async stream with Ollama API")
        loggable_params = {k: v for k, v in stream_params.items() if k != 'messages'}
        logger.debug(f"Ollama Stream parameters (excluding messages): {loggable_params}")

        try:
            stream = await client.chat.completions.create(**stream_params)
            logger.info("Ollama stream connection established")
            async for chunk in stream:
                if callback:
                    await callback(chunk)
                yield chunk
            logger.info("Ollama stream completed successfully")

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Could not connect to Ollama at {ollama_base_url}. Is it running? Error: {e}", exc_info=True)
            raise
        except Exception as stream_error:
            logger.error(f"Error during async streaming with Ollama: {str(stream_error)}", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Error in async stream_ollama: {str(e)}", exc_info=True)
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
        anakin_api_key = os.getenv('ANAKIN_API_KEY')
        if not anakin_api_key:
            logger.error("ANAKIN_API_KEY is not configured in settings")
            raise ValueError("ANAKIN_API_KEY is not configured")

        # Get app_id from settings if not provided
        if not app_id:
            app_id = os.getenv('ANAKIN_APP_ID', None)
            if not app_id:
                logger.error("ANAKIN_APP_ID is not configured in settings and not provided")
                raise ValueError("ANAKIN_APP_ID is not configured")

        anakin_base_url = "https://api.anakin.ai"
        api_version = os.getenv('ANAKIN_API_VERSION', '2024-05-06')

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


async def get_json(
    prompt: str,
    model: str,
    response_schema: Dict[str, Any],
    schema_name: str,
    system_prompt: Optional[str] = None,
    should_use_ollama: bool = False,
    schema_description: Optional[str] = "Extract content based on the provided schema."
):
    """
    Get a JSON response from a model.
    For Ollama, it uses simple JSON mode, as not all models support tools.
    For other providers (OpenRouter), it uses instructor for tool-calling.
    This function is now asynchronous.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if should_use_ollama:
        # For Ollama, we'll guide it to produce JSON in the prompt and use JSON mode.
        prompt_with_schema = f"""
        {prompt}

        You must respond in a valid JSON format that adheres to the following schema:
        {json.dumps(response_schema, indent=2)}
        """
        messages.append({"role": "user", "content": prompt_with_schema})
        
        client = AsyncOpenAI(
            base_url=os.getenv('OLLAMA_BASE_URL', "http://localhost:11434/v1"),
            api_key='ollama',
        )
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse JSON from Ollama response: {e}", exc_info=True)
            logger.debug(f"Ollama response content: {response.choices[0].message.content}")
            raise ValueError("Could not get a valid JSON from Ollama.")
    else:
        # For OpenRouter and other providers, we use the more robust tool-calling.
        messages.append({"role": "user", "content": prompt})

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            logger.error("OPENROUTER_API_KEY is not configured in settings")
            raise ValueError("OPENROUTER_API_KEY is not configured")
        
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        client = instructor.patch(client, mode=instructor.Mode.TOOLS)

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": schema_name,
                        "description": schema_description,
                        "parameters": {
                            "type": "object",
                            "properties": response_schema.get("properties", {}),
                            "required": response_schema.get("required", list(response_schema.get("properties", {}).keys())),
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": schema_name}},
        )
        
        try:
            arguments_str = response.choices[0].message.tool_calls[0].function.arguments
            return json.loads(arguments_str)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse tool call from LLM response: {e}", exc_info=True)
            logger.debug(f"LLM response object: {response}")
            raise ValueError("Could not extract a valid JSON tool call from the LLM response.")

