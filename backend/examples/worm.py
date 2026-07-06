# --------------------------------------------------------------
# Worm Example – Gym Swimmer-v5 wrapper for training and visualization
# --------------------------------------------------------------
# This version follows the Gymnasium MuJoCo Swimmer environment. It keeps
# the websocket interface identical to other examples so the frontend can drive
# training / inference while the underlying physics and reward match the
# reference implementation documented at
# https://gymnasium.farama.org/environments/mujoco/swimmer/
#
# Key points:
#   • Action space  : Box(-1, 1, (2,))  – raw torques for 2 joints
#   • Observation   : 8-D vector (angles, velocities)
#   • Reward        : forward_reward - ctrl_cost (handled by Gym)
#   • Termination   : 1000 steps (handled by Gym)
#
# NOTE: This file purposefully avoids extra shaping or privileged information.

from typing import List, Dict, Any

import numpy as np
from fastapi import WebSocket

import gymnasium as gym
import mujoco

# -----------------------------------------------------------------------------------
# Environment wrapper (adds a small helper for Three.js visualisation)
# -----------------------------------------------------------------------------------


class WormEnvWrapper:
    """Thin wrapper around gym.make('Swimmer-v5') with helper for rendering state."""

    def __init__(self):
        # With n links, there are n-1 joints. Default is 3 links.
        self.env = gym.make(
            "Swimmer-v5", exclude_current_positions_from_observation=True
        )
        self.obs, _ = self.env.reset()

        # Keep reference to the underlying MuJoCo data for viz
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data

        # Cache body part names. Default swimmer has 'torso', 'mid', 'back'
        self.body_names = ["torso", "mid", "back"]

    def reset(self):
        self.obs, _ = self.env.reset()
        return self.obs

    def step(self, action: np.ndarray):
        self.obs, reward, done, truncated, _ = self.env.step(action)
        # Gymnasium returns both done & truncated; treat either as episode end
        return self.obs, reward, done or truncated

    # ------------------------------------------------------------------
    # Helper for frontend rendering – returns pose info for each segment.
    # ------------------------------------------------------------------
    def get_state_for_viz(self) -> Dict[str, Any]:
        segments = []
        for name in self.body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()

            # Find the first capsule geom associated with this body to get its size
            geom_id = -1
            for i in range(self.model.ngeom):
                if (
                    self.model.geom_bodyid[i] == body_id
                    and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE
                ):
                    geom_id = i
                    break

            # Default size, fallback if no capsule geom is found
            size = np.array([0.1, 0.1])
            if geom_id != -1:
                # MuJoCo capsule size is [radius, half-height]
                size = self.model.geom_size[geom_id].copy()

            segments.append(
                {
                    "name": name,
                    "pos": pos.tolist(),
                    "quat": quat.tolist(),
                    "size": size.tolist(),
                }
            )
        return {"segments": segments}


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_worm(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "worm")


def infer_action_worm(obs: List[float], model_filename: str | None = None):
    from mlagents.websocket_training import predict_policy_action

    return predict_policy_action("worm", obs, model_filename)


async def run_worm(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "worm",
        WormEnvWrapper,
        model_filename=model_filename,
        action_transform=lambda action: np.asarray(action, dtype=np.float32),
    )
