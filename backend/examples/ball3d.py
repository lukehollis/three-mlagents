from typing import List

import numpy as np
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
    np.array([TILT_DELTA, 0.0]),  # +x
    np.array([-TILT_DELTA, 0.0]),  # −x
    np.array([0.0, TILT_DELTA]),  # +z
    np.array([0.0, -TILT_DELTA]),  # −z
    np.array([0.0, 0.0]),  # no-op
]
NUM_ACTIONS = len(ACTION_DELTAS)


class Ball3DEnv:
    """A lightweight physics approximation of the ML-Agents 3DBall task."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Platform rotation (x, z) in radians – start with a random small tilt
        self.rot = np.random.uniform(-MAX_TILT * 0.5, MAX_TILT * 0.5, size=2).astype(
            np.float32
        )

        # Ball position relative to platform centre – random within half size
        self.pos = np.random.uniform(-1.5, 1.5, size=2).astype(np.float32)

        # Ball velocity (x, z) – give it a small random push so corrective control is needed
        self.vel = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(
            [
                self.rot[0],
                self.rot[1],
                self.pos[0],
                self.pos[1],
                self.vel[0],
                self.vel[1],
            ],
            dtype=np.float32,
        )

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
        off_platform = (abs(self.pos[0]) > PLATFORM_HALF_SIZE) or (
            abs(self.pos[1]) > PLATFORM_HALF_SIZE
        )
        timeout = self.steps >= MAX_STEPS_PER_EP
        done = off_platform or timeout

        # Reward scheme: +0.1 per step alive, −1 when failure, +1 bonus if survived full episode
        center_dist = np.linalg.norm(self.pos)  # 0 at centre, grows as ball drifts
        reward = 1.0 - center_dist / PLATFORM_HALF_SIZE  # ∈ (-∞, 1], highest at centre
        if done:
            reward = -1.0
            if timeout and not off_platform:
                reward = +1.0

        dist_penalty = -0.02 * np.linalg.norm(self.pos)
        reward += dist_penalty

        return self._get_obs(), reward, done


# ---------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# ---------------------------------------------------------------------------------


async def train_ball3d(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(
        websocket,
        "ball3d",
        total_timesteps=150_000,
        algorithm="ppo",
        progress_freq=2_000,
    )


def infer_action_ball3d(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("ball3d", obs, model_filename)
