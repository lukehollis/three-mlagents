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
from examples.crawler import train_ant, infer_action_ant, run_ant
from examples.worm import train_worm, infer_action_worm, run_worm
from examples.worm import WormEnvWrapper
from examples.brick_break import train_brick_break, run_brick_break, BrickBreakEnv
from examples.food_collector import train_food_collector, run_food_collector, FoodCollectorEnv
from examples.bicycle import train_bicycle, run_bicycle, BicycleEnv
from examples.glider import train_glider, run_glider, GliderEnv
from examples.minefarm import run_minefarm, MineFarmEnv

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
@app.websocket("/ws/ant")
async def websocket_ant(ws: WebSocket):
    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_ant(ws)
        elif cmd == "run":
            await run_ant(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # raw 111-D observation
            act_vec = infer_action_ant(obs)
            await ws.send_json({"type": "action", "action": act_vec})


# WebSocket endpoint for Worm
@app.websocket("/ws/worm")
async def websocket_worm(ws: WebSocket):
    await ws.accept()
    # Send initial state so the frontend can render the worm before training or running
    preview_env = WormEnvWrapper()
    preview_state = preview_env.get_state_for_viz()
    await ws.send_json({"type": "state", "state": preview_state, "episode": 0})
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_worm(ws)
        elif cmd == "run":
            await run_worm(ws)
        elif cmd == "inference":
            obs = data.get("obs", [])  # raw 8-D observation
            act_vec = infer_action_worm(obs)
            await ws.send_json({"type": "action", "action": act_vec})


# WebSocket endpoint for BrickBreak
@app.websocket("/ws/brickbreak")
async def websocket_brick_break(websocket: WebSocket):
    await websocket.accept()
    preview_env = BrickBreakEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": preview_state, "episode": 0})
    try:
        while True:
            data = await websocket.receive_json()
            if data['cmd'] == 'train':
                await train_brick_break(websocket)
            elif data['cmd'] == 'run':
                await run_brick_break(websocket)
    except Exception as e:
        print(f"BrickBreak websocket disconnected: {e}")


# WebSocket endpoint for FoodCollector
@app.websocket("/ws/foodcollector")
async def websocket_food_collector(websocket: WebSocket):
    await websocket.accept()
    preview_env = FoodCollectorEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": preview_state, "episode": 0})
    try:
        while True:
            data = await websocket.receive_json()
            if data['cmd'] == 'train':
                await train_food_collector(websocket)
            elif data['cmd'] == 'run':
                await run_food_collector(websocket)
    except Exception as e:
        print(f"FoodCollector websocket disconnected: {e}")


# WebSocket endpoint for Bicycle
@app.websocket("/ws/bicycle")
async def websocket_bicycle(websocket: WebSocket):
    await websocket.accept()
    preview_env = BicycleEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": preview_state, "episode": 0})
    try:
        while True:
            data = await websocket.receive_json()
            if data['cmd'] == 'train':
                await train_bicycle(websocket)
            elif data['cmd'] == 'run':
                await run_bicycle(websocket)
    except Exception as e:
        print(f"Bicycle websocket disconnected: {e}")


# WebSocket endpoint for Glider
@app.websocket("/ws/glider")
async def websocket_glider(websocket: WebSocket):
    await websocket.accept()
    preview_env = GliderEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": preview_state, "episode": 0})
    try:
        while True:
            data = await websocket.receive_json()
            if data['cmd'] == 'train':
                await train_glider(websocket)
            elif data['cmd'] == 'run':
                await run_glider(websocket)
    except Exception as e:
        print(f"Glider websocket disconnected: {e}")


# WebSocket endpoint for MineFarm
@app.websocket("/ws/minefarm")
async def websocket_minefarm(websocket: WebSocket):
    await websocket.accept()
    preview_env = MineFarmEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": preview_state})
    try:
        while True:
            data = await websocket.receive_json()
            if data['cmd'] == 'run':
                await run_minefarm(websocket)
    except Exception as e:
        print(f"MineFarm websocket disconnected: {e}")

