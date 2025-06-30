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

# --------------------------------------------------------------
# Simple Push-Block Environment --------------------------------
# --------------------------------------------------------------

DEFAULT_GRID_SIZE = 6  # N x N grid
MAX_STEPS_PER_EP = 100

# Actions: 0 stay, 1 up, 2 down, 3 left, 4 right
ACTION_DELTAS: List[Tuple[int, int]] = [
    (0, 0),   # stay
    (0, 1),   # up (+z)
    (0, -1),  # down (−z)
    (-1, 0),  # left (−x)
    (1, 0),   # right (+x)
]
NUM_ACTIONS = len(ACTION_DELTAS)

# Observation: relative vectors (agent→box, box→goal)
OBS_SIZE = 4


class PushEnv:
    """Minimal push-block environment.

    The agent must push a movable box into a goal area located on the
    top row of the grid. Only cardinal moves are allowed.
    """

    def __init__(self, grid_size: int = DEFAULT_GRID_SIZE):
        self.grid_size = grid_size
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(cells)
        self.agent_pos = cells[0]
        self.box_pos = cells[1]

        # Choose a random goal cell on the top row (y == grid_size − 1)
        goal_x = np.random.randint(0, self.grid_size)
        self.goal_pos = (goal_x, self.grid_size - 1)

        self.steps = 0
        return self._get_obs()

    # ----------------------------------------------------------
    def _get_obs(self):
        # Relative vectors normalised to [−1,1]
        dx_ab = (self.box_pos[0] - self.agent_pos[0]) / max(1, self.grid_size - 1)
        dy_ab = (self.box_pos[1] - self.agent_pos[1]) / max(1, self.grid_size - 1)
        dx_bg = (self.goal_pos[0] - self.box_pos[0]) / max(1, self.grid_size - 1)
        dy_bg = (self.goal_pos[1] - self.box_pos[1]) / max(1, self.grid_size - 1)
        return np.array([dx_ab, dy_ab, dx_bg, dy_bg], dtype=np.float32)

    # ----------------------------------------------------------
    def step(self, action_idx: int):
        dx, dy = ACTION_DELTAS[action_idx]
        new_agent_x = int(np.clip(self.agent_pos[0] + dx, 0, self.grid_size - 1))
        new_agent_y = int(np.clip(self.agent_pos[1] + dy, 0, self.grid_size - 1))

        new_box_x, new_box_y = self.box_pos
        reward = -0.01
        done = False

        # Attempt push if agent moves into box
        if (new_agent_x, new_agent_y) == self.box_pos:
            tentative_box_x = self.box_pos[0] + dx
            tentative_box_y = self.box_pos[1] + dy
            # If push is within bounds, move box
            if 0 <= tentative_box_x < self.grid_size and 0 <= tentative_box_y < self.grid_size:
                new_box_x, new_box_y = tentative_box_x, tentative_box_y
            else:
                # Invalid push – cancel agent movement
                new_agent_x, new_agent_y = self.agent_pos
        # Update positions
        self.agent_pos = (new_agent_x, new_agent_y)
        self.box_pos = (new_box_x, new_box_y)
        self.steps += 1

        # Check goal achievement
        if self.box_pos == self.goal_pos:
            reward = 1.0
            done = True

        if self.steps >= MAX_STEPS_PER_EP:
            done = True

        return self._get_obs(), reward, done

# --------------------------------------------------------------
# Q-network -----------------------------------------------------
# --------------------------------------------------------------

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

# --------------------------------------------------------------
# Training logic ------------------------------------------------
# --------------------------------------------------------------

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


async def train_push(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"push_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[PushEnv] = [PushEnv() for _ in range(12)]

    net = QNet()
    target_net = QNet()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.95
    epsilon = 1.0
    episodes = 1500
    target_update_freq = 100

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

            next_obs_list: List[np.ndarray] = []
            rewards: List[float] = []
            dones: List[bool] = []
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

            active_idxs = [i for i, d in enumerate(done_flags) if not d]
            if active_idxs:
                obs_tensor = torch.tensor([obs_list[i] for i in active_idxs], dtype=torch.float32)
                next_obs_tensor = torch.tensor([next_obs_list[i] for i in active_idxs], dtype=torch.float32)
                actions_tensor = torch.tensor([actions[i] for i in active_idxs], dtype=torch.long)
                rewards_tensor = torch.tensor([rewards[i] for i in active_idxs], dtype=torch.float32)
                dones_tensor = torch.tensor([dones[i] for i in active_idxs], dtype=torch.float32)

                q_pred = net(obs_tensor).gather(1, actions_tensor.view(-1, 1)).squeeze()
                with torch.no_grad():
                    q_next_max = target_net(next_obs_tensor).max(dim=1).values
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

            if step_counter % target_update_freq == 0:
                target_net.load_state_dict(net.state_dict())

            env0 = envs[0]
            await websocket.send_json({
                "type": "train_step",
                "state": {
                    "agentX": int(env0.agent_pos[0]),
                    "agentY": int(env0.agent_pos[1]),
                    "boxX": int(env0.box_pos[0]),
                    "boxY": int(env0.box_pos[1]),
                    "goalX": int(env0.goal_pos[0]),
                    "goalY": int(env0.goal_pos[1]),
                    "gridSize": env0.grid_size,
                },
                "episode": ep + 1,
            })

            if step_counter % 20 == 0:
                await asyncio.sleep(0)

            if step_counter >= MAX_STEPS_PER_EP:
                break

        epsilon = max(0.05, epsilon * 0.995)

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

# --------------------------------------------------------------
# Inference -----------------------------------------------------
# --------------------------------------------------------------

_ort_sessions_push = {}


def infer_action_push(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort

    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("push_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No trained push policy files available")
        files.sort(reverse=True)
        model_filename = files[0]

    if model_filename not in _ort_sessions_push:
        model_path = os.path.join(POLICIES_DIR, model_filename)
        _ort_sessions_push[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inp = np.array([obs], dtype=np.float32)
    outputs = _ort_sessions_push[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 