# --------------------------------------------------------------
# Crawler Example – Gym Ant-v5 wrapper for training and visualization
# --------------------------------------------------------------
# This version follows the Gymnasium MuJoCo Ant environment exactly. It keeps
# the websocket interface identical to crawler.py so the frontend can drive
# training / inference while the underlying physics and reward match the
# reference implementation documented at
# https://mgoulao.github.io/gym-docs/environments/mujoco/ant/
#
# Key points:
#   • Action space  : Box(-1, 1, (8,))  – raw torques
#   • Observation   : 111-D vector as in Gym Ant (x,y excluded)
#   • Reward        : healthy + forward − ctrl_cost − contact_cost (handled by Gym)
#   • Termination   : unhealthy or 1000 steps (handled by Gym)
#
# NOTE: This file purposefully avoids extra shaping or privileged information.

from typing import List

import numpy as np
from fastapi import WebSocket

import gymnasium as gym
import mujoco  # for mj_name2id helper

# -----------------------------------------------------------------------------------
# Environment wrapper (adds a small helper for Three.js visualisation)
# -----------------------------------------------------------------------------------


class AntEnvWrapper:
    """Thin wrapper around gym.make('Ant-v5') with helper for rendering state."""

    def __init__(self):
        self.env = gym.make("Ant-v5", exclude_current_positions_from_observation=True)
        self.obs, _ = self.env.reset()

        # Keep reference to the underlying MuJoCo data for viz
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        # Cache torso body index (API differs between mujoco-py and mujoco >=2.3)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action: np.ndarray):
        self.obs, reward, done, truncated, _ = self.env.step(action)
        # Gymnasium returns both done & truncated; treat either as episode end
        return self.obs, reward, done or truncated

    # ------------------------------------------------------------------
    # Helper for frontend rendering – returns minimal pose information.
    # ------------------------------------------------------------------
    def get_state_for_viz(self):
        # Base position and orientation (quaternion) from MuJoCo data
        pos = self.data.xpos[self.torso_id].copy()
        quat = self.data.xquat[self.torso_id].copy()

        # Joint names for the 8 actuated joints, in the order the frontend expects
        # (front-left, front-right, back-left, back-right). This ensures the
        # visualization matches the physical model.
        joint_names = [
            "hip_1",
            "angle_1",
            "hip_2",
            "angle_2",
            "hip_3",
            "angle_3",
            "hip_4",
            "angle_4",
        ]

        joint_qpos = []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            addr = self.model.jnt_qposadr[jid]
            joint_qpos.append(float(self.data.qpos[addr]))

        return {
            "basePos": pos.tolist(),
            "baseOri": quat.tolist(),
            "jointAngles": joint_qpos,
        }


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_ant(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "ant")


def infer_action_ant(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_policy_action

    return predict_policy_action("ant", obs, model_filename)


async def run_ant(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "ant",
        AntEnvWrapper,
        model_filename=model_filename,
        action_transform=lambda action: np.asarray(action, dtype=np.float32),
    )
