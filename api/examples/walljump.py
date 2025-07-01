# --------------------------------------------------------------
# Wall Jump Example - simplified for browser RL demo
# --------------------------------------------------------------

import os
import asyncio
from datetime import datetime
import uuid
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket

# --------------------------------------------------------------
# Environment ---------------------------------------------------
# --------------------------------------------------------------

WIDTH = 20  # 1-D track length
MAX_STEPS = 150

# Actions: 0 stay, 1 forward, 2 backward, 3 jump
NUM_ACTIONS = 4
ACTION_DELTAS = [0, 1, -1, 1]  # jump also moves forward

OBS_SIZE = 4  # [dx_goal, dx_wall, wall_height, on_ground]


class WallJumpEnv:
    """Minimal 1-D wall-jump environment.

    The agent moves along the x-axis starting at 0 aiming to reach WIDTH-1.
    A wall may be present at x == WALL_X. If present (height == 1), the
    agent must jump to cross the wall. Jumps last JUMP_DURATION steps.
    """

    WALL_X = 10
    JUMP_DURATION = 3

    def __init__(self):
        self.width = WIDTH
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        self.agent_x: int = 0
        self.in_air: int = 0  # remaining steps in air (0 means grounded)
        self.wall_height: int = int(np.random.rand() < 0.7)  # 70% wall present
        self.steps: int = 0
        return self._obs()

    # ----------------------------------------------------------
    def _obs(self):
        dx_goal = (self.width - 1 - self.agent_x) / (self.width - 1)
        dx_wall = (self.WALL_X - self.agent_x) / (self.width - 1)
        wall_h = float(self.wall_height)
        on_ground = 1.0 if self.in_air == 0 else 0.0
        return np.array([dx_goal, dx_wall, wall_h, on_ground], dtype=np.float32)

    # ----------------------------------------------------------
    def step(self, action_idx: int):
        assert 0 <= action_idx < NUM_ACTIONS

        reward = -0.01  # step penalty
        done = False

        just_jumped = False
        if action_idx == 3 and self.in_air == 0:
            self.in_air = self.JUMP_DURATION
            just_jumped = True

        # Determine proposed move
        dx = ACTION_DELTAS[action_idx]
        proposed_x = int(np.clip(self.agent_x + dx, 0, self.width - 1))

        # Block movement by wall if not jumping
        crossing_wall = (
            (self.agent_x < self.WALL_X <= proposed_x) or
            (proposed_x < self.WALL_X <= self.agent_x)
        )
        if crossing_wall and self.wall_height == 1 and self.in_air == 0:
            proposed_x = self.agent_x  # cannot cross
            reward -= 0.02  # slight penalty for hitting wall

        # Penalise jumping when not needed (wall not immediately ahead)
        if just_jumped and not crossing_wall and abs(self.WALL_X - self.agent_x) > 1:
            reward -= 0.03

        self.agent_x = proposed_x

        # Update air timer
        if self.in_air > 0:
            self.in_air -= 1

        # Success condition
        if self.agent_x == self.width - 1:
            reward = 1.0
            done = True

        self.steps += 1
        if self.steps >= MAX_STEPS:
            done = True

        return self._obs(), reward, done


# --------------------------------------------------------------
# Q-network -----------------------------------------------------
# --------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, NUM_ACTIONS)

    def forward(self, x):  # type: ignore[override]
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


# --------------------------------------------------------------
# Training ------------------------------------------------------
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


async def train_walljump(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"walljump_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[WallJumpEnv] = [WallJumpEnv() for _ in range(16)]

    net = QNet()
    target_net = QNet()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.95
    epsilon = 1.0
    episodes = 3000
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
                actions_tensor = torch.tensor([int(actions[i]) for i in active_idxs], dtype=torch.long)
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

            # Visualise first env every few steps
            if step_counter % 5 == 0:
                env0 = envs[0]
                await websocket.send_json({
                    "type": "train_step",
                    "state": {
                        "agentX": int(env0.agent_x),
                        "goalX": env0.width - 1,
                        "wallX": env0.WALL_X,
                        "wallPresent": int(env0.wall_height),
                        "gridSize": env0.width,
                    },
                    "episode": ep + 1,
                })

            if step_counter % 20 == 0:
                await asyncio.sleep(0)

            if step_counter >= MAX_STEPS:
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

_ort_sessions_wj: dict[str, "onnxruntime.InferenceSession"] = {}


def infer_action_walljump(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort

    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("walljump_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No trained walljump policy files available")
        files.sort(reverse=True)
        model_filename = files[0]

    if model_filename not in _ort_sessions_wj:
        model_path = os.path.join(POLICIES_DIR, model_filename)
        _ort_sessions_wj[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inp = np.array([obs], dtype=np.float32)
    outputs = _ort_sessions_wj[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 