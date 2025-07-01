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

TRACK_LEN = 30  # 1-D track length (along +x)
MAX_STEPS = 200

#           0     1         2          3
# Actions: stay, turn_left, turn_right, move_forward
NUM_ACTIONS = 4

OBS_SIZE = 5  # [dx_goal, cos_heading, vel_norm, target_speed_norm, sin_heading]


class CrawlerEnv:
    """Very light-weight 1-D crawler environment.

    The agent starts at x == 0 with a random target speed in (0.1, 1.0].
    Heading (yaw) is continuous in [−π, π). The goal is to move towards
    +x direction without falling (modelled here as exceeding |heading| > π/2).
    Reward each step is geometric (product) of:
        speed_reward  ∈ [0,1] – match between current speed and target speed
        heading_reward ∈ [0,1] – alignment of heading with +x direction
    """

    TURN_DELTA = 0.25  # radians per left/right action
    FORWARD_STEP = 1.0  # units per forward action
    MAX_SPEED = FORWARD_STEP  # used for normalisation

    def __init__(self):
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        self.x: float = 0.0
        self.heading: float = 0.0  # 0 rad points towards +x (goal)
        self.prev_x: float = 0.0
        self.steps: int = 0
        self.target_speed: float = float(np.random.uniform(0.1, 1.0))
        return self._obs()

    # ----------------------------------------------------------
    def _obs(self):
        dx_goal = (TRACK_LEN - 1 - self.x) / max(1.0, TRACK_LEN - 1)
        vel = np.clip(self.x - self.prev_x, -self.MAX_SPEED, self.MAX_SPEED)
        vel_norm = (vel + self.MAX_SPEED) / (2 * self.MAX_SPEED)  # map to [0,1]
        tgt_speed_norm = self.target_speed / 1.0  # already ≤1
        return np.array([
            dx_goal,
            np.cos(self.heading),
            vel_norm,
            tgt_speed_norm,
            np.sin(self.heading),
        ], dtype=np.float32)

    # ----------------------------------------------------------
    def step(self, action_idx: int):
        assert 0 <= action_idx < NUM_ACTIONS

        if action_idx == 1:  # turn left
            self.heading += self.TURN_DELTA
        elif action_idx == 2:  # turn right
            self.heading -= self.TURN_DELTA
        elif action_idx == 3:  # move forward along heading
            self.prev_x = self.x
            self.x += self.FORWARD_STEP * float(np.cos(self.heading))
        # action 0 is stay (no changes)

        # Clamp heading into [−π, π)
        if self.heading >= np.pi:
            self.heading -= 2 * np.pi
        if self.heading < -np.pi:
            self.heading += 2 * np.pi

        # Speed (|Δx|) for reward calc
        curr_speed = abs(self.x - self.prev_x)

        # Rewards ------------------------------------------------
        speed_reward = 1.0 - abs(curr_speed - self.target_speed) / max(1e-5, self.target_speed)
        speed_reward = np.clip(speed_reward, 0.0, 1.0)
        heading_reward = (np.cos(self.heading) + 1.0) * 0.5  # 1 when heading==0 rad
        step_reward = speed_reward * heading_reward  # geometric

        done = False
        if self.x >= TRACK_LEN - 1:
            step_reward = 1.0  # big reward for success
            done = True

        self.steps += 1
        if self.steps >= MAX_STEPS:
            done = True

        obs = self._obs()
        return obs, float(step_reward), done


# --------------------------------------------------------------
# Q-network (discrete actions) ----------------------------------
# --------------------------------------------------------------


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, NUM_ACTIONS)

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


async def train_crawler(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"crawler_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[CrawlerEnv] = [CrawlerEnv() for _ in range(16)]

    net = QNet()
    target_net = QNet()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.97
    epsilon = 1.0
    episodes = 4000
    target_update_freq = 120

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
                        "agentX": float(env0.x),
                        "heading": float(env0.heading),
                        "gridSize": TRACK_LEN,
                        "goalX": TRACK_LEN - 1,
                    },
                    "episode": ep + 1,
                })

            if step_counter % 20 == 0:
                await asyncio.sleep(0)

            if step_counter >= MAX_STEPS:
                break

        epsilon = max(0.05, epsilon * 0.994)

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

_ort_sessions_cr: dict[str, "onnxruntime.InferenceSession"] = {}


def infer_action_crawler(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort

    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("crawler_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No trained crawler policy files available")
        files.sort(reverse=True)
        model_filename = files[0]

    if model_filename not in _ort_sessions_cr:
        model_path = os.path.join(POLICIES_DIR, model_filename)
        _ort_sessions_cr[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    inp = np.array([obs], dtype=np.float32)
    outputs = _ort_sessions_cr[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 