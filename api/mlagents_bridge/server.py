
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import json
import logging
import queue
from typing import Dict

logger = logging.getLogger("ThreeJSServer")

def create_server(env_map: Dict[str, "ThreeJSEnv"]):
    """
    Creates a FastAPI app with a websocket endpoint that connects to ThreeJSEnvs.
    env_map: dict of channel_id -> ThreeJSEnv
    """
    app = FastAPI()

    @app.websocket("/ws/mlagents")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        logger.info("Client connected to /ws/mlagents")
        
        # We assume single channel/single environment for now, or use query param?
        # Let's assume default channel
        env = env_map.get("default")
        
        if not env:
            await ws.close(code=1000, reason="No environment found")
            return

        # Connect queues
        # We need async wrappers or run in thread
        
        try:
            while True:
                # 1. Check for outbound messages from Python (Poll non-blocking?)
                # Since we are in async loop, we can't easily wait on a blocking Queue.
                # But we can use run_in_executor or just poll with small sleep.
                
                # Check outbound (Python -> Browser)
                try:
                    while True:
                         msg = env.outbound_queue.get_nowait()
                         await ws.send_json(msg)
                except queue.Empty:
                    pass
                
                # 2. Check for inbound messages from Browser
                # receive_json is awaitable
                try:
                    # We use wait_for to allow polling both directions
                    data = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                    # Put into inbound queue for Python to read
                    env.inbound_queue.put(data)
                except asyncio.TimeoutError:
                    pass # continue loop
                
                await asyncio.sleep(0.001)

        except WebSocketDisconnect:
            logger.info("Client disconnected")
            env.close()

    return app
