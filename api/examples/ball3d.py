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

# ---------------------------------------------------------------------------------
# Simplified 3DBall environment ----------------------------------------------------
# ---------------------------------------------------------------------------------

G = 9.81  # gravitational constant (m/s^2) – only used for rough accel scaling
DT = 0.02  # physics time-step (seconds)
MAX_STEPS_PER_EP = 200  # terminate episode after this many physics steps

# Bounds for the square platform on which the ball must stay
PLATFORM_HALF_SIZE = 3.0  # ball falls when |x| or |z| > 3

# Bounds for platform rotation (approximately ±25° in radians)
MAX_TILT = np.deg2rad(25.0)
TILT_DELTA = np.deg2rad(3.0)  # amount each discrete action tilts the platform

# Observation indices for convenience
#   0: rotX, 1: rotZ, 2: ballPosX, 3: ballPosZ, 4: ballVelX, 5: ballVelZ
OBS_SIZE = 6

# Discrete action mapping (5 actions)
#  0: tilt +x  (rotate platform around Z-axis positive)
#  1: tilt −x  (rotate platform around Z-axis negative)
#  2: tilt +z  (rotate platform around X-axis positive)
#  3: tilt −z  (rotate platform around X-axis negative)
#  4: no-op
ACTION_DELTAS = [
    np.array([ TILT_DELTA,  0.0]),  # +x
    np.array([-TILT_DELTA,  0.0]),  # −x
    np.array([ 0.0,  TILT_DELTA]),  # +z
    np.array([ 0.0, -TILT_DELTA]),  # −z
    np.array([ 0.0,  0.0]),         # no-op
]
NUM_ACTIONS = len(ACTION_DELTAS)


class Ball3DEnv:
    """A lightweight physics approximation of the ML-Agents 3DBall task."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Platform rotation (x, z) in radians – start with a random small tilt
        self.rot = np.random.uniform(-MAX_TILT * 0.5, MAX_TILT * 0.5, size=2).astype(np.float32)

        # Ball position relative to platform centre – random within half size
        self.pos = np.random.uniform(-1.5, 1.5, size=2).astype(np.float32)

        # Ball velocity (x, z) – give it a small random push so corrective control is needed
        self.vel = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([
            self.rot[0],
            self.rot[1],
            self.pos[0],
            self.pos[1],
            self.vel[0],
            self.vel[1],
        ], dtype=np.float32)

    def step(self, action_idx: int):
        # Apply platform tilt change, clip to limits
        delta = ACTION_DELTAS[action_idx]
        self.rot += delta
        self.rot = np.clip(self.rot, -MAX_TILT, MAX_TILT)

        # Compute acceleration of ball due to gravity projected onto tilted plane
        acc_x = G * np.sin(self.rot[0])
        acc_z = G * np.sin(self.rot[1])
        self.vel[0] += acc_x * DT
        self.vel[1] += acc_z * DT

        # Dampen velocity slightly (friction / rolling resistance)
        self.vel *= 0.98

        # Integrate position
        self.pos += self.vel * DT

        # Increment step counter
        self.steps += 1

        # Check termination conditions – ball fell off or time limit reached
        off_platform = (abs(self.pos[0]) > PLATFORM_HALF_SIZE) or (abs(self.pos[1]) > PLATFORM_HALF_SIZE)
        timeout = self.steps >= MAX_STEPS_PER_EP
        done = off_platform or timeout

        # Reward scheme: +0.1 per step alive, −1 when failure, +1 bonus if survived full episode
        center_dist = np.linalg.norm(self.pos)          # 0 at centre, grows as ball drifts
        reward = 1.0 - center_dist / PLATFORM_HALF_SIZE # ∈ (-∞, 1], highest at centre
        if done:
            reward = -1.0
            if timeout and not off_platform:
                reward = +1.0

        dist_penalty = -0.02 * np.linalg.norm(self.pos)
        reward += dist_penalty

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------------------
# Neural network & RL algorithm (simple Q-learning with discretised actions) -------
# ---------------------------------------------------------------------------------

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, NUM_ACTIONS)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)


# ---------------------------------------------------------------------------------
# Training entry point exposed to FastAPI websocket --------------------------------
# ---------------------------------------------------------------------------------

POLICIES_DIR = "policies"

def _export_model_onnx(model: nn.Module, path: str):
    dummy_input = torch.zeros((1, OBS_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


async def train_ball3d(websocket: WebSocket):
    # Ensure directory for saved policies exists
    os.makedirs(POLICIES_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"ball3d_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    # Create vectorised environments (12 instances to match UI layout)
    envs: List[Ball3DEnv] = [Ball3DEnv() for _ in range(12)]

    net = QNet()
    target_net = QNet()
    target_net.load_state_dict(net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(net.parameters(), lr=3e-4)
    gamma = 0.99
    epsilon = 1.0
    # Train long enough for the agent to discover Z-axis tilt actions as in the official Unity
    # implementation (see Ball3DAgent.cs / Ball3DHardAgent.cs in the ML-Agents repo). 1500
    # episodes still completes quickly because the per-step sleep has been removed.
    episodes = 1500
    target_update_freq = 100  # how many environment steps between target network updates

    for ep in range(episodes):
        # Reset all envs
        obs_list = [env.reset() for env in envs]
        done_flags = [False] * len(envs)
        total_reward = 0.0
        ep_loss_accum = 0.0
        step_counter = 0

        while not all(done_flags):
            obs_batch = torch.tensor(obs_list, dtype=torch.float32)
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                actions = np.random.randint(0, NUM_ACTIONS, size=len(envs))
            else:
                with torch.no_grad():
                    qvals = net(obs_batch)
                actions = torch.argmax(qvals, dim=1).cpu().numpy()

            # Step each env
            next_obs_list = []
            rewards = []
            dones = []
            for idx, env in enumerate(envs):
                if done_flags[idx]:
                    # Already finished – keep state
                    next_obs_list.append(obs_list[idx])
                    rewards.append(0.0)
                    dones.append(True)
                    continue
                nobs, rew, dn = env.step(int(actions[idx]))
                next_obs_list.append(nobs)
                rewards.append(rew)
                dones.append(dn)

            # Q-learning update per env (independent)
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            next_obs_tensor = torch.tensor(next_obs_list, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(dones, dtype=torch.float32)

            q_pred = net(obs_tensor).gather(1, actions_tensor.view(-1, 1)).squeeze()
            with torch.no_grad():
                q_next_max = target_net(next_obs_tensor).max(dim=1).values
                q_target = rewards_tensor + gamma * (1.0 - dones_tensor) * q_next_max
            loss = ((q_pred - q_target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Sync target network at specified frequency for more stable learning
            if step_counter % target_update_freq == 0:
                target_net.load_state_dict(net.state_dict())

            ep_loss_accum += float(loss.item())
            step_counter += 1
            total_reward += float(np.sum(rewards))

            obs_list = next_obs_list
            done_flags = dones

            # Stream current state to frontend for one of the envs (choose env 0)
            env0 = envs[0]
            await websocket.send_json({
                "type": "train_step",
                "state": {
                    "rotX": float(env0.rot[0]),
                    "rotZ": float(env0.rot[1]),
                    "ballX": float(env0.pos[0]),
                    "ballZ": float(env0.pos[1]),
                },
                "episode": ep + 1,
            })

            # Yield control occasionally so FastAPI event loop stays responsive
            if step_counter % 25 == 0:
                await asyncio.sleep(0)

            # Break loop if steps too many (safety)
            if step_counter >= MAX_STEPS_PER_EP:
                break

        # Epsilon annealing – slower decay so exploration lasts across the longer run
        epsilon = max(0.05, epsilon * 0.99)

        # Send progress summary every 10 episodes
        if (ep + 1) % 10 == 0:
            avg_loss = ep_loss_accum / max(1, step_counter)
            await websocket.send_json({
                "type": "progress",
                "episode": ep + 1,
                "reward": round(total_reward / len(envs), 3),
                "loss": round(avg_loss, 5),
            })

    # Export trained model
    _export_model_onnx(net, model_path)

    await websocket.send_json({
        "type": "trained",
        "file_url": f"/policies/{model_filename}",
        "model_filename": model_filename,
        "timestamp": timestamp,
        "session_uuid": session_uuid,
    })


# ---------------------------------------------------------------------------------
# Inference helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------------

_ort_sessions_ball3d = {}


def infer_action_ball3d(obs: List[float], model_filename: str = None):
    """Run the ONNX policy to obtain an action index (0-4)."""
    # Lazy import to avoid dependency if unused
    import onnxruntime as ort

    # Resolve model filename (most recent if None)
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("ball3d_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No trained ball3d policy files available")
        files.sort(reverse=True)
        model_filename = files[0]

    if model_filename not in _ort_sessions_ball3d:
        model_path = os.path.join(POLICIES_DIR, model_filename)
        _ort_sessions_ball3d[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Ensure obs is length 6 (rotX, rotZ, posX, posZ, velX, velZ)
    if len(obs) == 4:
        obs = list(obs) + [0.0, 0.0]  # backward-compat: assume zero velocity if not provided
    inp = np.array([obs], dtype=np.float32)
    outputs = _ort_sessions_ball3d[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 