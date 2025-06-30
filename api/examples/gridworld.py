# ---------------------------------------------------------------------------------
# GridWorld environment and Q-learning trainer -------------------------------------
# ---------------------------------------------------------------------------------

import os
import asyncio
from datetime import datetime
import uuid
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket

# ---------------------------------------------------------------------------------
# Simplified discrete GridWorld ----------------------------------------------------
# ---------------------------------------------------------------------------------

DEFAULT_GRID_SIZE = 5  # N x N grid
MAX_STEPS_PER_EP = 100

# Action mapping
# 0: no-op, 1: up (+z), 2: down (−z), 3: left (−x), 4: right (+x)
ACTION_DELTAS: List[Tuple[int, int]] = [
    (0, 0),   # stay
    (0, 1),   # up
    (0, -1),  # down
    (-1, 0),  # left
    (1, 0),   # right
]
NUM_ACTIONS = len(ACTION_DELTAS)

# Observation is 4-dim float vector:
#  [dx_to_goal, dy_to_goal, goal_one_hot_0, goal_one_hot_1]
OBS_SIZE = 4


class GridWorldEnv:
    """Minimal multi-goal GridWorld with one agent and two goal types."""

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        # Random positions – ensure they are all unique
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(all_cells)
        self.agent_pos = all_cells[0]
        self.green_goals = [all_cells[1]]  # "plus" goals
        self.red_goals = [all_cells[2]]    # "ex" goals
        # Randomly assign current target goal type
        self.current_goal_type = np.random.choice([0, 1])  # 0 = green, 1 = red
        self.steps = 0
        return self._get_obs()

    # -------------------------------------------------------------------------
    def _get_obs(self):
        # Vector from agent to the *nearest* target goal of the required type
        if self.current_goal_type == 0:
            goal = self.green_goals[0]
        else:
            goal = self.red_goals[0]
        dx = (goal[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        dy = (goal[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)
        one_hot_goal = [1.0, 0.0] if self.current_goal_type == 0 else [0.0, 1.0]
        return np.array([dx, dy, *one_hot_goal], dtype=np.float32)

    # -------------------------------------------------------------------------
    def step(self, action_idx: int):
        delta = ACTION_DELTAS[action_idx]
        new_x = int(np.clip(self.agent_pos[0] + delta[0], 0, self.grid_size - 1))
        new_y = int(np.clip(self.agent_pos[1] + delta[1], 0, self.grid_size - 1))
        self.agent_pos = (new_x, new_y)
        self.steps += 1

        # Base step penalty
        reward = -0.01
        done = False

        # Check for goal collision
        if self.agent_pos in self.green_goals:
            if self.current_goal_type == 0:
                reward = 1.0
            else:
                reward = -1.0
            done = True
        elif self.agent_pos in self.red_goals:
            if self.current_goal_type == 1:
                reward = 1.0
            else:
                reward = -1.0
            done = True

        if self.steps >= MAX_STEPS_PER_EP:
            done = True

        return self._get_obs(), reward, done

# ---------------------------------------------------------------------------------
# Neural network – simple MLP Q-network -------------------------------------------
# ---------------------------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)

# ---------------------------------------------------------------------------------
# Training exposed to FastAPI via websocket ---------------------------------------
# ---------------------------------------------------------------------------------

POLICIES_DIR = "policies"


def _export_model_onnx(model: nn.Module, path: str):
    dummy = torch.zeros((1, OBS_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


async def train_gridworld(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"gridworld_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[GridWorldEnv] = [GridWorldEnv() for _ in range(12)]

    net = QNet()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.95
    epsilon = 1.0
    episodes = 100

    for ep in range(episodes):
        obs_list = [env.reset() for env in envs]
        done_flags = [False] * len(envs)
        total_reward = 0.0
        ep_loss_accum = 0.0
        step_counter = 0

        while not all(done_flags):
            obs_batch = torch.tensor(obs_list, dtype=torch.float32)
            if np.random.rand() < epsilon:
                actions = np.random.randint(0, NUM_ACTIONS, size=len(envs))
            else:
                with torch.no_grad():
                    qvals = net(obs_batch)
                actions = torch.argmax(qvals, dim=1).cpu().numpy()

            next_obs_list = []
            rewards = []
            dones = []
            for idx, env in enumerate(envs):
                if done_flags[idx]:
                    next_obs_list.append(obs_list[idx])
                    rewards.append(0.0)
                    dones.append(True)
                    continue
                nobs, rew, dn = env.step(int(actions[idx]))
                next_obs_list.append(nobs)
                rewards.append(rew)
                dones.append(dn)

            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            next_obs_tensor = torch.tensor(next_obs_list, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)

            q_pred = net(obs_tensor).gather(1, actions_tensor.view(-1, 1)).squeeze()
            with torch.no_grad():
                q_next_max = net(next_obs_tensor).max(dim=1).values
                q_target = rewards_tensor + gamma * (1.0 - dones_tensor) * q_next_max
            loss = ((q_pred - q_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss_accum += float(loss.item())
            step_counter += 1
            total_reward += float(np.sum(rewards))

            obs_list = next_obs_list
            done_flags = dones

            # stream first env state for visualization
            env0 = envs[0]
            await websocket.send_json({
                "type": "train_step",
                "state": {
                    "agentX": int(env0.agent_pos[0]),
                    "agentY": int(env0.agent_pos[1]),
                    "gridSize": env0.grid_size,
                    "greenGoals": env0.green_goals,
                    "redGoals": env0.red_goals,
                    "currentGoalType": int(env0.current_goal_type),
                },
                "episode": ep + 1,
            })

            await asyncio.sleep(0.01)

            if step_counter >= MAX_STEPS_PER_EP:
                break

        epsilon = max(0.05, epsilon * 0.99)

        if (ep + 1) % 10 == 0:
            avg_loss = ep_loss_accum / max(1, step_counter)
            await websocket.send_json({
                "type": "progress",
                "episode": ep + 1,
                "reward": round(total_reward / len(envs), 3),
                "loss": round(avg_loss, 5),
            })

    _export_model_onnx(net, model_path)

    await websocket.send_json({
        "type": "trained",
        "file_url": f"/policies/{model_filename}",
        "model_filename": model_filename,
        "timestamp": timestamp,
        "session_uuid": session_uuid,
    })

# ---------------------------------------------------------------------------------
# Inference -----------------------------------------------------------------------
# ---------------------------------------------------------------------------------

_ort_sessions_grid = {}


def infer_action_gridworld(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort

    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("gridworld_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No trained gridworld policy files available")
        files.sort(reverse=True)
        model_filename = files[0]

    if model_filename not in _ort_sessions_grid:
        model_path = os.path.join(POLICIES_DIR, model_filename)
        _ort_sessions_grid[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inp = np.array([obs], dtype=np.float32)
    outputs = _ort_sessions_grid[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 