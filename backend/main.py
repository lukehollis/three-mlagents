from fastapi import FastAPI, WebSocket, HTTPException
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import asdict
from contextlib import asynccontextmanager
import json
import os
import asyncio
import subprocess
import time
import requests
from fastapi.staticfiles import StaticFiles

from examples.basic import train_basic, infer_action
from examples.ball3d import train_ball3d, infer_action_ball3d
from examples.gridworld import train_gridworld, infer_action_gridworld
from examples.push import train_push, infer_action_push
from examples.walljump import train_walljump, infer_action_walljump
from examples.brick_break import train_brick_break, run_brick_break, BrickBreakEnv
from examples.food_collector import FoodCollectorEnv
from examples.bicycle import train_bicycle, run_bicycle, BicycleEnv
from examples.glider import train_glider, run_glider, GliderEnv
from examples.astrodynamics import (
    train_astrodynamics,
    run_astrodynamics,
    run_simulation,
)
from examples.labyrinth import train_labyrinth, run_labyrinth, LabyrinthEnv
from examples.fish import FishEnv
from starlette.websockets import WebSocketState
import logging
from examples.kraken import train_pirate_ship, run_pirate_ship, PirateShipEnv
from mlagents.registry import get_task, list_task_cards
from mlagents.training import (
    TrainConfig,
    evaluate_model,
    latest_model_filename,
    train_task,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _send_websocket_error_and_close(
    websocket: WebSocket,
    *,
    feature: str,
    exc: Exception,
) -> None:
    message = f"{feature} backend is unavailable: {exc}"
    logger.warning(message)
    if websocket.application_state != WebSocketState.CONNECTED:
        await websocket.accept()
    await websocket.send_json({"type": "error", "message": message})
    await websocket.close(code=1000)


async def _send_unstandardized_training_error(websocket: WebSocket, task_id: str):
    task = get_task(task_id)
    await websocket.send_json(
        {
            "type": "error",
            "message": (
                f"{task.title} training is disabled in this backend because "
                f"'{task_id}' is registered as {task.interface}/{task.status}. "
                "Use a PettingZoo-compatible implementation before making "
                "paper-grade multi-agent or LLM/RL training claims."
            ),
        }
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_tensorboard_server()
    try:
        yield
    finally:
        stop_tensorboard_server()


app = FastAPI(title="Three ML-Agents Backend", lifespan=lifespan)

# Add CORS middleware to allow frontend to access API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class TaskTrainRequest(BaseModel):
    algorithm: str | None = None
    total_timesteps: int | None = None
    seed: int = 1
    n_envs: int | None = None
    eval_episodes: int | None = None
    eval_freq: int = 10_000
    run_name: str | None = None


class TaskEvaluateRequest(BaseModel):
    model_filename: str | None = None
    episodes: int | None = None
    deterministic: bool = True
    seed: int = 10_001


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks(include_roadmap: bool = True):
    return {"tasks": list_task_cards(include_roadmap=include_roadmap)}


@app.get("/tasks/{task_id}")
def task_detail(task_id: str):
    try:
        return get_task(task_id).card()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/tasks/{task_id}/train")
async def train_task_endpoint(task_id: str, request: TaskTrainRequest):
    try:
        result = await asyncio.to_thread(
            train_task,
            TrainConfig(
                task_id=task_id,
                total_timesteps=request.total_timesteps,
                algorithm=request.algorithm,
                seed=request.seed,
                n_envs=request.n_envs,
                eval_episodes=request.eval_episodes,
                eval_freq=request.eval_freq,
                run_name=request.run_name,
                verbose=0,
            ),
        )
        return asdict(result)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tasks/{task_id}/evaluate")
async def evaluate_task_endpoint(task_id: str, request: TaskEvaluateRequest):
    try:
        model_filename = request.model_filename or latest_model_filename(task_id)
        return await asyncio.to_thread(
            evaluate_model,
            task_id,
            model_filename,
            episodes=request.episodes,
            deterministic=request.deterministic,
            seed=request.seed,
        )
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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

# TensorBoard server management
tensorboard_process = None
TENSORBOARD_PORT = 6006


def start_tensorboard_server():
    """Start TensorBoard server for astrodynamics logs."""
    global tensorboard_process
    if tensorboard_process is not None:
        return  # Already running

    log_dir = "./astrodynamics_tensorboard"
    if not os.path.exists(log_dir):
        return  # No logs to serve

    try:
        tensorboard_process = subprocess.Popen(
            [
                "tensorboard",
                "--logdir",
                log_dir,
                "--port",
                str(TENSORBOARD_PORT),
                "--host",
                "0.0.0.0",
                "--reload_interval",
                "1",
            ]
        )
        print(f"TensorBoard server started on port {TENSORBOARD_PORT}")
    except Exception as e:
        print(f"Failed to start TensorBoard server: {e}")


def stop_tensorboard_server():
    """Stop TensorBoard server."""
    global tensorboard_process
    if tensorboard_process is not None:
        tensorboard_process.terminate()
        tensorboard_process = None
        print("TensorBoard server stopped")


@app.get("/tensorboard/astrodynamics")
async def get_tensorboard():
    """Redirect to TensorBoard interface for astrodynamics."""
    start_tensorboard_server()

    # Wait a moment for server to start
    for _ in range(10):  # Wait up to 1 second
        try:
            response = requests.get(f"http://localhost:{TENSORBOARD_PORT}", timeout=0.1)
            if response.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)

    return RedirectResponse(url=f"http://localhost:{TENSORBOARD_PORT}")


@app.get("/tensorboard/status")
async def tensorboard_status():
    """Check if TensorBoard server is running."""
    global tensorboard_process
    if tensorboard_process is None:
        return {"running": False}

    try:
        requests.get(f"http://localhost:{TENSORBOARD_PORT}", timeout=1)
        return {"running": True, "port": TENSORBOARD_PORT}
    except requests.RequestException:
        return {"running": False}


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
    from examples.crawler import infer_action_ant, run_ant, train_ant

    await ws.accept()
    async for message in ws.iter_text():
        data = json.loads(message)
        cmd = data.get("cmd")
        if cmd == "train":
            await train_ant(ws)
        elif cmd == "run":
            await run_ant(ws, data.get("model_filename"))
        elif cmd == "inference":
            obs = data.get("obs", [])  # raw 111-D observation
            act_vec = infer_action_ant(obs)
            await ws.send_json({"type": "action", "action": act_vec})


# WebSocket endpoint for Worm
@app.websocket("/ws/worm")
async def websocket_worm(ws: WebSocket):
    from examples.worm import WormEnvWrapper, infer_action_worm, run_worm, train_worm

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
            await run_worm(ws, data.get("model_filename"))
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
            if data["cmd"] == "train":
                await train_brick_break(websocket)
            elif data["cmd"] == "run":
                await run_brick_break(websocket, data.get("model_filename"))
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
            if data["cmd"] == "train":
                await _send_unstandardized_training_error(websocket, "foodcollector")
            elif data["cmd"] == "run":
                await _send_unstandardized_training_error(websocket, "foodcollector")
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
            if data["cmd"] == "train":
                await train_bicycle(websocket)
            elif data["cmd"] == "run":
                await run_bicycle(websocket, data.get("model_filename"))
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
            if data["cmd"] == "train":
                await train_glider(websocket)
            elif data["cmd"] == "run":
                await run_glider(websocket, data.get("model_filename"))
    except Exception as e:
        print(f"Glider websocket disconnected: {e}")


# WebSocket endpoint for Astrodynamics
@app.websocket("/ws/astrodynamics")
async def websocket_astrodynamics(websocket: WebSocket):
    await websocket.accept()
    # Start with the physics-only simulation
    active_task = asyncio.create_task(run_simulation(websocket))

    try:
        while True:
            message = await websocket.receive_json()
            cmd = message.get("cmd")

            if cmd:
                # Cancel the currently active task
                if active_task:
                    active_task.cancel()
                    try:
                        await active_task
                    except asyncio.CancelledError:
                        pass  # Expected

                # Handle the new command
                if cmd == "train":
                    # Training is a blocking operation
                    await train_astrodynamics(websocket)
                    # After training, revert to the default simulation
                    active_task = asyncio.create_task(run_simulation(websocket))

                elif cmd == "run":
                    # Running with a model is a non-blocking background task
                    model_filename = message.get("model_filename")
                    active_task = asyncio.create_task(
                        run_astrodynamics(websocket, model_filename)
                    )

    except WebSocketDisconnect:
        print("Client disconnected from astrodynamics")
    finally:
        # Clean up the active task when the connection closes
        if "active_task" in locals() and active_task and not active_task.done():
            active_task.cancel()


@app.websocket("/ws/labyrinth")
async def websocket_labyrinth(websocket: WebSocket):
    await websocket.accept()

    # Send an initial static state
    env = LabyrinthEnv(training_mode=False)
    state = env.get_state_for_viz()
    await websocket.send_json({"type": "state", "state": state})

    active_task = None

    try:
        while True:
            message = await websocket.receive_json()
            cmd = message.get("cmd")

            if cmd:
                if active_task:
                    active_task.cancel()
                    try:
                        await active_task
                    except asyncio.CancelledError:
                        pass

                if cmd == "train":
                    active_task = asyncio.create_task(train_labyrinth(websocket))

                elif cmd == "run":
                    model_filename = message.get("model_filename")
                    active_task = asyncio.create_task(
                        run_labyrinth(websocket, model_filename)
                    )

    except WebSocketDisconnect:
        print("WebSocket client disconnected from labyrinth endpoint")
        if active_task:
            print("Cancelling active training/run task due to disconnection")
            active_task.cancel()
            try:
                await active_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        print(f"Error in labyrinth WebSocket: {e}")
        if active_task:
            active_task.cancel()
            try:
                await active_task
            except asyncio.CancelledError:
                pass


# WebSocket endpoint for MineCraft
@app.websocket("/ws/minecraft")
async def websocket_endpoint_minecraft(websocket: WebSocket):
    await websocket.accept()
    try:
        from examples.minecraft import MineCraftEnv
    except ModuleNotFoundError as exc:
        await _send_websocket_error_and_close(websocket, feature="Minecraft", exc=exc)
        return

    env = None
    run_task = None
    try:
        # On first connection, create environment and send initial state
        env = MineCraftEnv()
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "init", "state": state})

        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                await _send_unstandardized_training_error(websocket, "minecraft")
            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "minecraft")
            elif cmd == "stop":
                if run_task:
                    run_task.cancel()
                    run_task = None
            elif cmd == "reset":
                if run_task:
                    run_task.cancel()
                    run_task = None
                env = MineCraftEnv()
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "reset", "state": state})

    except WebSocketDisconnect:
        print("Client disconnected from minecraft ws")
        if run_task:
            run_task.cancel()
    except Exception as e:
        logger.error(f"Error in minecraft websocket: {e}", exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_json({"type": "error", "message": str(e)})


# WebSocket endpoint for Fish
@app.websocket("/ws/fish")
async def websocket_fish(websocket: WebSocket):
    await websocket.accept()
    env = FishEnv()
    initial_state = env.get_state_for_viz()
    initial_state["agents"] = initial_state.pop("agents")  # Rename for frontend
    await websocket.send_json({"type": "init", "state": initial_state})

    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                await _send_unstandardized_training_error(websocket, "fish")
            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "fish")
    except WebSocketDisconnect:
        print("Fish websocket disconnected")


# WebSocket endpoint for Intersection
@app.websocket("/ws/intersection")
async def websocket_intersection(websocket: WebSocket):
    await websocket.accept()
    try:
        from examples.intersection import (
            MultiVehicleEnv as IntersectionEnv,
        )
    except ModuleNotFoundError as exc:
        await _send_websocket_error_and_close(
            websocket, feature="Intersection", exc=exc
        )
        return

    env = IntersectionEnv()
    initial_state = env.get_state_for_viz()
    await websocket.send_json({"type": "init", "state": initial_state})

    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                await _send_unstandardized_training_error(websocket, "intersection")
            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "intersection")

    except Exception as e:
        print(f"Intersection websocket disconnected: {e}")


# WebSocket endpoint for SelfDrivingCar
@app.websocket("/ws/self_driving_car")
async def websocket_self_driving_car(websocket: WebSocket):
    await websocket.accept()
    try:
        from examples.self_driving_car import (
            SelfDrivingCarEnv,
        )
    except ModuleNotFoundError as exc:
        await _send_websocket_error_and_close(
            websocket, feature="Self-driving car", exc=exc
        )
        return

    env = SelfDrivingCarEnv()
    run_task = None

    # Send initial state on connect
    initial_state = env.get_state_for_viz()
    await websocket.send_json({"type": "init", "state": initial_state})

    try:
        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                if run_task:
                    run_task.cancel()
                    run_task = None
                await _send_unstandardized_training_error(websocket, "self-driving-car")

            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "self-driving-car")

            elif cmd == "stop":
                if run_task:
                    run_task.cancel()
                    run_task = None
    except WebSocketDisconnect:
        print("Client disconnected from self-driving-car ws")
        if run_task:
            run_task.cancel()


@app.websocket("/ws/simcity")
async def websocket_endpoint_simcity(websocket: WebSocket):
    await websocket.accept()
    try:
        from examples.simcity import SimCityEnv
    except ModuleNotFoundError as exc:
        await _send_websocket_error_and_close(websocket, feature="SimCity", exc=exc)
        return

    env = None
    run_task = None
    try:
        # On first connection, create environment and send initial state
        env = SimCityEnv()
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "init", "state": state})

        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                await _send_unstandardized_training_error(websocket, "simcity")
            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "simcity")
            elif cmd == "stop":
                if run_task:
                    run_task.cancel()
                    run_task = None
            elif cmd == "reset":
                if run_task:
                    run_task.cancel()
                    run_task = None
                env = SimCityEnv()
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "reset", "state": state})

    except WebSocketDisconnect:
        print("Client disconnected from simcity ws")
        if run_task:
            run_task.cancel()
    except Exception as e:
        logger.error(f"Error in simcity websocket: {e}", exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_json({"type": "error", "message": str(e)})


@app.websocket("/ws/simcity_deckgl")
async def websocket_endpoint_simcity_deckgl(websocket: WebSocket):
    await websocket.accept()
    try:
        from examples.simcity_deckgl import (
            SimCityEnv as SimCityDeckGLEnv,
        )
    except ModuleNotFoundError as exc:
        await _send_websocket_error_and_close(
            websocket, feature="SimCity Deck.gl", exc=exc
        )
        return

    env = None
    run_task = None
    try:
        # On first connection, create environment and send initial state
        env = SimCityDeckGLEnv()
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "init", "state": state})

        while True:
            data = await websocket.receive_json()
            cmd = data.get("cmd")

            if cmd == "train":
                await _send_unstandardized_training_error(websocket, "simcity")
            elif cmd == "run":
                await _send_unstandardized_training_error(websocket, "simcity")
            elif cmd == "stop":
                if run_task:
                    run_task.cancel()
                    run_task = None
            elif cmd == "reset":
                if run_task:
                    run_task.cancel()
                    run_task = None
                env = SimCityDeckGLEnv()
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "reset", "state": state})

    except WebSocketDisconnect:
        print("Client disconnected from simcity_deckgl ws")
        if run_task:
            run_task.cancel()
    except Exception as e:
        logger.error(f"Error in simcity_deckgl websocket: {e}", exc_info=True)
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.send_json({"type": "error", "message": str(e)})


@app.websocket("/ws/kraken")
async def websocket_kraken(websocket: WebSocket):
    await websocket.accept()
    preview_env = PirateShipEnv()
    preview_state = preview_env.get_state_for_viz()
    await websocket.send_json({"type": "init", "state": preview_state})
    try:
        while True:
            data = await websocket.receive_json()
            if data["cmd"] == "train":
                await train_pirate_ship(websocket)
            elif data["cmd"] == "run":
                await run_pirate_ship(websocket, data.get("model_filename"))
    except WebSocketDisconnect:
        print("Kraken websocket disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
