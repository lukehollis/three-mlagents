import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any
from collections import deque
import random

import asyncio
import numpy as np
import torch
import torch.nn as nn
from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import NatureCNN


ACTION_SIZE = 4  # 0=up, 1=down, 2=left, 3=right
POLICIES_DIR = "policies"
# BATCH_SIZE = 8192 - Handled by SB3 n_steps
# MINI_BATCH = 512 - Handled by SB3 batch_size
# EPOCHS = 10 - Handled by SB3 n_epochs
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 1e-4
MAX_STEPS_PER_EPISODE = 2500
EPISODES = 5000
TOTAL_TIMESTEPS = MAX_STEPS_PER_EPISODE * EPISODES
LABYRINTH_WIDTH = 129
LABYRINTH_HEIGHT = 65

# -----------------------------------------------------------------------------------
# Labyrinth Environment
# -----------------------------------------------------------------------------------

class LabyrinthEnv(gym.Env):
    """A 2D ASCII labyrinth environment with a pursuer (Minotaur) and an exit."""
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, training_mode: bool = True):
        super().__init__()
        self.width = LABYRINTH_WIDTH
        self.height = LABYRINTH_HEIGHT
        self.grid = np.full((self.height, self.width), '#')
        self.training_mode = training_mode

        self.action_space = spaces.Discrete(ACTION_SIZE)
        
        # Observation space is the grid as uint8 image [0,255] for NatureCNN
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.height, self.width, 1), 
            dtype=np.uint8
        )
        
        # Episode tracking
        self.total_episode_reward = 0.0
        self.reset()

    def _generate_labyrinth(self):
        # Iterative maze generation to handle large sizes without recursion limits.
        grid = np.full((self.height, self.width), '#', dtype='<U1')
        
        def is_valid(y, x):
            return 1 <= y < self.height - 1 and 1 <= x < self.width - 1

        stack = []
        start_y, start_x = 1, 1
        grid[start_y, start_x] = ' '
        stack.append((start_y, start_x))

        while stack:
            y, x = stack[-1]
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)
            
            carved_new_path = False
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                nwy, nwx = y + dy // 2, x + dx // 2
                
                if is_valid(ny, nx) and grid[ny, nx] == '#':
                    grid[nwy, nwx] = ' '
                    grid[ny, nx] = ' '
                    stack.append((ny, nx))
                    carved_new_path = True
                    break
            
            if not carved_new_path:
                stack.pop()
        return grid
    
    def _get_random_empty_cell(self):
        while True:
            y = random.randint(1, self.height - 2)
            x = random.randint(1, self.width - 2)
            if self.grid[y, x] == ' ':
                return (y, x)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self._generate_labyrinth()
        
        self.theseus_pos = self._get_random_empty_cell()
        
        self.minotaur_pos = self._get_random_empty_cell()
        while np.linalg.norm(np.array(self.theseus_pos) - np.array(self.minotaur_pos)) < 40:
            self.minotaur_pos = self._get_random_empty_cell()

        self.exit_pos = self._get_random_empty_cell()
        while np.linalg.norm(np.array(self.theseus_pos) - np.array(self.exit_pos)) < 40:
            self.exit_pos = self._get_random_empty_cell()
            
        self.grid[self.exit_pos] = 'E'
        self.steps = 0
        self.minotaur_turn_counter = 0
        self.total_episode_reward = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1
        py, px = self.theseus_pos
        ny, nx = py, px

        if action == 0: ny -= 1   # Up
        elif action == 1: ny += 1 # Down
        elif action == 2: nx -= 1 # Left
        elif action == 3: nx += 1 # Right

        reward = -0.05  # Small time penalty

        # Penalty for hitting a wall
        if self.grid[ny, nx] == '#':
            reward -= 0.1
        else:
            # Reward for moving closer to the exit
            dist_to_exit_prev = abs(py - self.exit_pos[0]) + abs(px - self.exit_pos[1])
            dist_to_exit_new = abs(ny - self.exit_pos[0]) + abs(nx - self.exit_pos[1])
            reward += 0.05 * (dist_to_exit_prev - dist_to_exit_new)
            self.theseus_pos = (ny, nx)

        # Penalty for moving closer to the Minotaur
        dist_to_mino_prev = abs(py - self.minotaur_pos[0]) + abs(px - self.minotaur_pos[1])
        dist_to_mino_new = abs(self.theseus_pos[0] - self.minotaur_pos[0]) + abs(self.theseus_pos[1] - self.minotaur_pos[1])
        reward -= 0.02 * (dist_to_mino_prev - dist_to_mino_new)

        # Move Minotaur every two steps
        self.minotaur_turn_counter += 1
        if self.minotaur_turn_counter % 2 == 0:
            self._move_minotaur()

        terminated = False
        truncated = False
        info = {}
        episode_end_reason = None
        
        if self.theseus_pos == self.exit_pos:
            reward = 200.0
            terminated = True
            episode_end_reason = "Reached the exit"
        elif self.theseus_pos == self.minotaur_pos:
            reward = -100.0
            terminated = True
            episode_end_reason = "Caught by Minotaur"
        elif self.steps >= MAX_STEPS_PER_EPISODE:
            reward = -20.0
            truncated = True
            episode_end_reason = "Timeout"

        self.total_episode_reward += reward

        if terminated or truncated:
            if episode_end_reason and self.training_mode:
                info["episode_end"] = {
                    "reason": episode_end_reason,
                    "steps": self.steps,
                    "total_reward": self.total_episode_reward,
                }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _move_minotaur(self):
        my, mx = self.minotaur_pos
        ty, tx = self.theseus_pos
        dy, dx = np.sign(ty - my), np.sign(tx - mx)
        
        # Smarter greedy move: try axis with largest distance first
        if abs(ty - my) > abs(tx - mx):
            if dy != 0 and self.grid[my + int(dy), mx] != '#':
                self.minotaur_pos = (my + int(dy), mx)
            elif dx != 0 and self.grid[my, mx + int(dx)] != '#':
                self.minotaur_pos = (my, mx + int(dx))
        else:
            if dx != 0 and self.grid[my, mx + int(dx)] != '#':
                self.minotaur_pos = (my, mx + int(dx))
            elif dy != 0 and self.grid[my + int(dy), mx] != '#':
                self.minotaur_pos = (my + int(dy), mx)

    def _get_obs(self):
        # Observation as uint8 image [0,255] for NatureCNN
        obs_grid = np.full((self.height, self.width, 1), 51, dtype=np.uint8) # Path ~0.2*255
        
        # Walls are 0
        wall_y, wall_x = np.where(self.grid == '#')
        obs_grid[wall_y, wall_x, 0] = 0
        
        # Exit is 255
        exit_y, exit_x = self.exit_pos
        obs_grid[exit_y, exit_x, 0] = 255

        # Theseus is ~204 (0.8*255)
        ty, tx = self.theseus_pos
        obs_grid[ty, tx, 0] = 204
        
        # Minotaur is ~102 (0.4*255)
        my, mx = self.minotaur_pos
        obs_grid[my, mx, 0] = 102
        
        return obs_grid

    def get_state_for_viz(self) -> Dict[str, Any]:
        grid_viz = self.grid.copy()
        grid_viz[self.theseus_pos] = 'T'
        grid_viz[self.minotaur_pos] = 'M'
        return {"grid": grid_viz.tolist(), "steps": self.steps}

# -----------------------------------------------------------------------------------
# Callbacks and training setup
# -----------------------------------------------------------------------------------

async def send_json_safely(websocket: WebSocket, payload: dict) -> bool:
    """Safely send JSON over a websocket, returning True on success, False on failure."""
    if websocket.application_state != WebSocketState.CONNECTED:
        print("Skipping send, websocket not connected.")
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, ConnectionClosedError) as e:
        print(f"Failed to send message: client disconnected. Reason: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while sending message: {e}")
        return False

class WebSocketCallback(BaseCallback):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop, verbose=0):
        super(WebSocketCallback, self).__init__(verbose)
        self.websocket = websocket
        self.loop = loop
        self.should_stop = False
        self.episode_count = 0
        self.vis_env_idx = 0
        self.step_counter = 0

    def _send_message_safely(self, payload: dict) -> bool:
        if self.should_stop or self.websocket.application_state != WebSocketState.CONNECTED:
            if not self.should_stop:
                print("WebSocket is not connected. Stopping training.")
                self.should_stop = True
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                send_json_safely(self.websocket, payload), self.loop
            )
            return future.result(timeout=1.0)
        except Exception as e:
            print(f"WebSocket send error: {e}. Stopping training.")
            self.should_stop = True
            return False

    def _on_step(self) -> bool:
        if self.should_stop: return False

        self.step_counter += 1
        self.vis_env_idx = self.step_counter % self.training_env.num_envs

        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode_end" in info:
                    self.episode_count += 1
                    ep_info = info["episode_end"]
                    print(
                        f"Episode {self.episode_count} (env {i}): {ep_info['reason']} "
                        f"in {ep_info['steps']} steps. "
                        f"Total reward: {ep_info['total_reward']:.2f}"
                    )
                    payload = {
                        "type": "episode_end",
                        "episode": self.episode_count,
                        "timestep": self.num_timesteps,
                        "reason": ep_info["reason"],
                        "steps": ep_info["steps"],
                        "reward": ep_info["total_reward"],
                    }
                    if not self._send_message_safely(payload): return False

        # Send state of one environment at every step for visualization
        try:
            states = self.training_env.env_method("get_state_for_viz", indices=[self.vis_env_idx])
            if states and states[0]:
                payload = {
                    "type": "train_step",
                    "state": states[0],
                    "episode": self.episode_count,
                    "timestep": self.num_timesteps,
                }
                if not self._send_message_safely(payload): return False
        except Exception as e:
            print(f"Error getting env state for viz: {e}")
        
        return not self.should_stop

    def _on_rollout_end(self) -> None:
        if self.should_stop: return
        
        reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
        loss = self.logger.name_to_value.get("train/loss")
        print(f"Rollout end. Timestep: {self.num_timesteps}, Mean Reward: {reward:.2f}, Loss: {loss:.4f}")

        payload = {
            "type": "progress",
            "episode": self.episode_count,
            "timestep": self.num_timesteps,
            "reward": float(reward) if reward is not None else None,
            "loss": float(loss) if loss is not None else None,
        }
        if not self._send_message_safely(payload):
            self.should_stop = True

    def stop_training(self):
        print("Training stop requested via callback.")
        self.should_stop = True

def _export_model_onnx(model: PPO, path: str):
    class ExportableModel(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self.policy.extract_features(obs)
            action_logits = self.policy.action_net(self.policy.mlp_extractor.policy_net(features))
            return action_logits

    exportable_model = ExportableModel(model.policy)
    exportable_model.eval()

    dummy_input = torch.randn(1, *model.observation_space.shape, device=model.device)

    torch.onnx.export(
        exportable_model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

# -----------------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------------

_training_task = None
_training_callback = None

async def train_labyrinth(websocket: WebSocket):
    global _training_task, _training_callback
    
    if _training_task and not _training_task.done():
        print("Stopping existing Labyrinth training...")
        if _training_callback: _training_callback.stop_training()
        _training_task.cancel()
        try: await _training_task
        except asyncio.CancelledError: pass

    print("Starting Labyrinth training...")
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs("labyrinth_tensorboard", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename_base = f"labyrinth_policy_{ts}_{session_uuid}"
    model_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.zip")
    onnx_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.onnx")
    
    loop = asyncio.get_running_loop()

    def train_model():
        try:
            print("Setting up vectorized environment for Labyrinth...")
            vec_env = make_vec_env(LabyrinthEnv, n_envs=16, env_kwargs=dict(training_mode=True))

            policy_kwargs = dict(
                features_extractor_class=NatureCNN,
                features_extractor_kwargs=dict(features_dim=256),
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            )

            print("Creating PPO model for Labyrinth...")
            model = PPO(
                "CnnPolicy",
                vec_env,
                verbose=1,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_EPS,
                ent_coef=ENT_COEF,
                learning_rate=LR,
                n_epochs=10,
                batch_size=256,
                n_steps=2048,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./labyrinth_tensorboard/"
            )

            print("Setting up WebSocket callback for Labyrinth...")
            websocket_callback = WebSocketCallback(websocket, loop)
            global _training_callback
            _training_callback = websocket_callback
            
            print("Starting Labyrinth training loop...")
            initial_payload = {"type": "training_started", "total_timesteps": TOTAL_TIMESTEPS}
            websocket_callback._send_message_safely(initial_payload)
            
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=websocket_callback,
                progress_bar=True
            )
            print("Training loop finished.")

            if not websocket_callback.should_stop and websocket.application_state == WebSocketState.CONNECTED:
                print("Saving Labyrinth model...")
                model.save(model_path)
                _export_model_onnx(model, onnx_path)
                print(f"Labyrinth model saved to {model_path} and {onnx_path}")
                return {"success": True, "model_filename_base": model_filename_base, "ts": ts, "session_uuid": session_uuid}
            else:
                reason = "Training stopped early by callback or disconnect."
                print(f"Not saving Labyrinth model: {reason}")
                return {"success": False, "reason": reason}
                
        except Exception as e:
            print(f"Labyrinth training error: {e}")
            import traceback
            traceback.print_exc()
            if _training_callback: _training_callback.stop_training()
            return {"success": False, "reason": str(e)}
    
    _training_task = asyncio.create_task(asyncio.to_thread(train_model))
    
    try:
        result = await _training_task
        print(f"Labyrinth training completed with result: {result}")
        
        if result["success"]:
            await send_json_safely(websocket, {
                "type": "trained",
                "file_url": f"/policies/{result['model_filename_base']}.zip",
                "model_filename": f"{result['model_filename_base']}.zip",
                "onnx_filename": f"{result['model_filename_base']}.onnx",
                "timestamp": result["ts"],
                "session_uuid": result["session_uuid"]
            })
        else:
            await send_json_safely(websocket, {
                "type": "training_error",
                "message": f"Training failed: {result.get('reason', 'Unknown error')}"
            })
    except asyncio.CancelledError:
        print("Labyrinth training task was cancelled.")
        if _training_callback: _training_callback.stop_training()
        await send_json_safely(websocket, {"type": "training_error", "message": "Training cancelled."})
    finally:
        _training_task = None
        _training_callback = None


async def run_simulation(websocket: WebSocket):
    env = LabyrinthEnv(training_mode=False)
    env.reset()
    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            state = env.get_state_for_viz()
            await send_json_safely(websocket, {"type": "state", "state": state})
            await asyncio.sleep(0.5)
            env.reset()
    except WebSocketDisconnect:
        print("Labyrinth simulation client disconnected.")


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_SB3_CACHE: dict[str, "PPO"] = {}

def infer_action_labyrinth(obs: np.ndarray, model_filename: str | None = None) -> int:
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("labyrinth_policy_") and f.endswith(".zip")]
        if not files:
            print("Warning: No labyrinth policy found. Returning random action.")
            return random.randint(0, ACTION_SIZE - 1)
        files.sort(reverse=True)
        model_filename = files[0]

    model_path = os.path.join(POLICIES_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_filename}' not found. Returning random action.")
        return random.randint(0, ACTION_SIZE - 1)
    
    if model_filename not in _SB3_CACHE:
        try:
            print(f"Loading Labyrinth SB3 model: {model_filename}")
            model = PPO.load(model_path)
            _SB3_CACHE[model_filename] = model
        except Exception as e:
            print(f"Error loading SB3 model '{model_filename}': {e}. Returning random action.")
            return random.randint(0, ACTION_SIZE - 1)
    
    model = _SB3_CACHE[model_filename]
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

async def run_labyrinth(websocket: WebSocket, model_filename: str | None = None):
    env = LabyrinthEnv(training_mode=False)
    episode = 0
    obs, _ = env.reset()
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_labyrinth(obs, model_filename)
        nobs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        state = env.get_state_for_viz()
        await send_json_safely(websocket, {"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.1)
        if done:
            episode += 1
            obs, _ = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0.01) 