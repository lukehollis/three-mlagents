from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import asyncio
from fastapi import WebSocket
from fastapi.staticfiles import StaticFiles

from examples.basic import train_basic, infer_action
from examples.ball3d import train_ball3d, infer_action_ball3d

app = FastAPI(title="ML-Agents API")

SMALL_GOAL = 7
LARGE_GOAL = 17
MIN_POS = 0
MAX_POS = 20
START_POS = 10


class StepRequest(BaseModel):
    action: int  # −1, 0, or +1
    position: int  # current agent position


class StepResponse(BaseModel):
    position: int
    reward: float
    done: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/basic/reset")
def reset_basic():
    return {"position": START_POS}


@app.post("/basic/step", response_model=StepResponse)
def step_basic(req: StepRequest):
    next_pos = req.position + req.action
    if next_pos < MIN_POS:
        next_pos = MIN_POS
    if next_pos > MAX_POS:
        next_pos = MAX_POS

    reward = -0.01
    done = False

    if next_pos == SMALL_GOAL:
        reward += 0.1
        done = True
    if next_pos == LARGE_GOAL:
        reward += 1.0
        done = True

    return StepResponse(position=next_pos, reward=reward, done=done)


# Ensure policies directory exists and is served as static files
os.makedirs("policies", exist_ok=True)
app.mount("/policies", StaticFiles(directory="policies"), name="policies")


# WebSocket endpoint


@app.websocket("/ws/basic")
async def websocket_basic(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_basic(ws)
        elif cmd == "inference":
            position = int(data.get("obs", 0))
            act_idx = await infer_action(position)
            await ws.send_json({"type": "action", "action": int(act_idx)})


# WebSocket endpoint for 3DBall


@app.websocket("/ws/ball3d")
async def websocket_ball3d(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_ball3d(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # expect list [rotX, rotZ, ballX, ballZ]
            act_idx = infer_action_ball3d(obs)
            await ws.send_json({"type": "action", "action": int(act_idx)}) 