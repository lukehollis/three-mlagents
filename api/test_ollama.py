#!/usr/bin/env python3
"""
Test script to debug Ollama and Gemma3n connectivity and performance issues.
This script helps isolate whether LLM timeout issues are in the minecraft code or the underlying service.
"""

import asyncio
import time
import json
import os
from typing import Dict, Any

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from services.llm import get_json, get_embedding

async def test_basic_connectivity():
    """Test basic connectivity to ollama service"""
    print("🔍 Testing basic Ollama connectivity...")
    
    simple_prompt = "Hello, please respond with just the word 'OK'"
    simple_schema = {
        "type": "object", 
        "properties": {
            "response": {"type": "string"}
        },
        "required": ["response"]
    }
    
    start_time = time.time()
    
    response = await get_json(
        prompt=simple_prompt,
        model="gemma3n:latest",
        response_schema=simple_schema,
        schema_name="basic_test",
        should_use_ollama=True
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"✅ Basic connectivity test completed in {elapsed:.2f}s")
    print(f"Response: {response}")
    return True

async def test_timeout_scenarios():
    """Test different timeout scenarios"""
    print("\n🔍 Testing timeout scenarios...")
    
    test_prompt = "Please provide a short response about reinforcement learning."
    test_schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"}
        },
        "required": ["content"]
    }
    
    timeouts = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    for timeout in timeouts:
        print(f"Testing with {timeout}s timeout...")
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                get_json(
                    prompt=test_prompt,
                    model="gemma3n:latest", 
                    response_schema=test_schema,
                    schema_name="timeout_test",
                    should_use_ollama=True
                ),
                timeout=timeout
            )
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"  ✅ Success in {elapsed:.2f}s: {response.get('content', '')[:50]}...")
            
        except asyncio.TimeoutError:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"  ❌ Timeout after {elapsed:.2f}s")
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"  ❌ Error after {elapsed:.2f}s: {e}")

async def test_concurrent_calls():
    """Test concurrent LLM calls like in minecraft.py"""
    print("\n🔍 Testing concurrent LLM calls...")
    
    num_concurrent = 5
    prompt_template = "You are agent {agent_id}. Respond with a brief message about mining in a 3D world."
    
    response_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        },
        "required": ["message"]
    }
    
    # Create concurrent tasks
    tasks = []
    for i in range(num_concurrent):
        prompt = prompt_template.format(agent_id=i)
        
        task = asyncio.create_task(
            asyncio.wait_for(
                get_json(
                    prompt=prompt,
                    model="gemma3n:latest",
                    response_schema=response_schema,
                    schema_name=f"concurrent_test_{i}",
                    should_use_ollama=True
                ),
                timeout=5.0
            )
        )
        tasks.append((i, task))
    
    print(f"Created {num_concurrent} concurrent tasks...")
    start_time = time.time()
    
    # Wait for tasks with a timeout
    done, pending = await asyncio.wait([task for _, task in tasks], timeout=10.0)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Completed {len(done)} tasks, {len(pending)} pending after {elapsed:.2f}s")
    
    # Process completed tasks
    for i, task in tasks:
        if task in done:
            try:
                result = task.result()
                print(f"  ✅ Agent {i}: {result.get('message', '')[:50]}...")
            except Exception as e:
                print(f"  ❌ Agent {i} failed: {e}")
        else:
            print(f"  ⏰ Agent {i}: Task still pending, cancelling...")
            task.cancel()

async def test_embedding_service():
    """Test the embedding service used for agent memory"""
    print("\n🔍 Testing embedding service...")
    
    test_texts = [
        "I found gold ore in the mine",
        "Agent communication about trading resources", 
        "Exploring the underground cave system"
    ]
    
    for text in test_texts:
        start_time = time.time()
        
        try:
            embedding = get_embedding(text)
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"  ✅ Embedded '{text[:30]}...' in {elapsed:.2f}s (shape: {embedding.shape})")
            
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"  ❌ Failed to embed '{text[:30]}...' after {elapsed:.2f}s: {e}")

async def test_complex_prompt():
    """Test with a complex prompt similar to what minecraft agents use"""
    print("\n🔍 Testing complex prompt (similar to minecraft agents)...")
    
    complex_prompt = """You are a mining agent in a 3D grid world. Your ID is 0.
Your current position is [x, y, z]: [32, 8, 32]. Y is the vertical axis.
Your inventory is {"grass": 0, "dirt": 2, "wood": 1, "stone": 5, "iron": 1, "gold": 0, "diamond": 0, "crystal": 0, "obsidian": 0}.
Your current goal is to collect 'diamond'.
Your current memory of recent events is summarized as: [0.12, -0.05, 0.33, 0.67, -0.23]...
Recent messages from other agents: [{"sender_id": 1, "message": "Found iron ore!", "step": 100}]
Crafting recipes available: {"stone_pickaxe": {"craft_time": 2, "value": 25, "recipe": {"stone": 3, "wood": 2}}}
Open trade offers: []

The world is 64x16x64. Resources are encoded as numbers. 0 is empty (air). The world is mostly solid stone underground; you must mine to find resources.
Resource map: 1: grass, 2: dirt, 3: wood, 4: stone, 5: iron, 6: gold, 7: diamond, 8: crystal, 9: obsidian.
You can see a 5x5x5 area around you. Your view:
[[[0,0,0],[0,4,0],[0,0,0]], [[4,4,4],[4,4,4],[4,4,4]], [[0,0,0],[0,4,0],[0,0,0]]]

Your available actions are "move", "mine", "talk", "craft", "offer", "accept_offer", or "wait".
- move: requires integer [x, y, z] coordinates for the next step.
- mine: requires integer [x, y, z] coordinates of an adjacent resource to mine.
- talk: requires an object with { "message": string, "recipient_id": integer (optional) }.

Based on your state, what is your next action?"""

    action_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["move", "mine", "talk", "craft", "offer", "accept_offer", "wait"]},
            "data": {"oneOf": [
                {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                {"type": "string"},
                {"type": "object"}, 
            ]}
        },
        "required": ["action", "data"]
    }
    
    start_time = time.time()
    
    try:
        response = await asyncio.wait_for(
            get_json(
                prompt=complex_prompt,
                model="gemma3n:latest",
                response_schema=action_schema,
                schema_name="complex_test",
                should_use_ollama=True
            ),
            timeout=8.0
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"  ✅ Complex prompt completed in {elapsed:.2f}s")
        print(f"  Response: {response}")
        
    except asyncio.TimeoutError:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"  ❌ Complex prompt timed out after {elapsed:.2f}s")
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"  ❌ Complex prompt failed after {elapsed:.2f}s: {e}")

async def main():
    """Run all tests"""
    print("🚀 Starting Ollama/Gemma3n Debug Tests")
    print("=" * 50)
    
    try:
        await test_basic_connectivity()
        await test_timeout_scenarios()
        await test_concurrent_calls()
        await test_embedding_service()
        await test_complex_prompt()
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Debug tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 