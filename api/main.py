from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import asyncio
from fastapi import WebSocket
from fastapi.staticfiles import StaticFiles

from examples.basic import train_basic, infer_action
from examples.ball3d import train_ball3d, infer_action_ball3d
from examples.gridworld import train_gridworld, infer_action_gridworld
from examples.push import train_push, infer_action_push
from examples.walljump import train_walljump, infer_action_walljump
from examples.crawler import train_crawler, infer_action_crawler, run_crawler

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


# WebSocket endpoint for GridWorld


@app.websocket("/ws/gridworld")
async def websocket_gridworld(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_gridworld(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # expect [dx, dy, g0, g1] (or any agreed)
            act_idx = infer_action_gridworld(obs)
            await ws.send_json({"type": "action", "action": int(act_idx)})


# WebSocket endpoint for Push environment


@app.websocket("/ws/push")
async def websocket_push(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_push(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # expect [dx_ab, dy_ab, dx_bg, dy_bg]
            act_idx = infer_action_push(obs)
            await ws.send_json({"type": "action", "action": int(act_idx)})


# WebSocket endpoint for Wall Jump


@app.websocket("/ws/walljump")
async def websocket_walljump(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_walljump(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # [dx_goal, dx_wall, wall_height, on_ground]
            act_idx = infer_action_walljump(obs)
            await ws.send_json({"type": "action", "action": int(act_idx)})


# WebSocket endpoint for Crawler


@app.websocket("/ws/crawler")
async def websocket_crawler(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_crawler(ws)
        elif cmd == "run":
            await run_crawler(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # expect [dx_goal, cos_head, vel_norm, tgt_speed_norm, sin_head]
            act_vec = infer_action_crawler(obs)
            # simple heuristic mapping continuous 20-D vector → discrete command for demo
            heading_signal = act_vec[0]  # first joint as proxy for turn
            forward_signal = abs(act_vec[2])  # third joint magnitude for move forward
            if forward_signal > 0.5:
                action_idx = 3  # move forward
            elif heading_signal > 0.3:
                action_idx = 1  # turn left
            elif heading_signal < -0.3:
                action_idx = 2  # turn right
            else:
                action_idx = 0  # no-op
            await ws.send_json({"type": "action", "action": action_idx}) 