from typing import List, Dict, Any

import numpy as np
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# Bicycle Environment
# -----------------------------------------------------------------------------------


class BicycleEnv:
    """A simple bicycle riding environment."""

    def __init__(self):
        # Physics constants
        self.g = 9.8
        self.h = 0.8  # height of center of mass
        self.L = 1.0  # wheelbase
        self.v = 5.0  # constant velocity
        self.dt = 0.02

        # State variables
        self.x = 0.0
        self.z = 0.0
        self.theta = 0.0  # heading angle
        self.phi = 0.0  # lean angle
        self.phi_dot = 0.0  # lean rate
        self.delta = 0.0  # steering angle

        self.max_phi = np.pi / 4  # fail if lean angle > 45 degrees
        self.max_delta = np.pi / 6  # max steering angle

        self.steps = 0

        self.goal_pos = np.zeros(2)
        self.dist_to_goal = 0.0

        self.reset()

    def reset(self):
        self.x = 0.0
        self.z = 0.0
        self.theta = 0.0
        # Start with a small random lean
        self.phi = np.random.uniform(-0.1, 0.1)
        self.phi_dot = np.random.uniform(-0.1, 0.1)
        self.delta = 0.0
        self.steps = 0

        # Set a new random goal in front of the agent
        goal_radius = np.random.uniform(15, 25)
        goal_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.goal_pos = np.array(
            [goal_radius * np.cos(goal_angle), goal_radius * np.sin(goal_angle)]
        )

        self.dist_to_goal = np.linalg.norm(self.goal_pos - np.array([self.x, self.z]))
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1

        # Action: 0=steer left, 1=no steer, 2=steer right
        steer_change = 0.0
        if action == 0:
            steer_change = -0.05
        elif action == 2:
            steer_change = 0.05

        self.delta += steer_change
        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)

        # Physics update
        # Simplified model: phi_ddot = gravity_torque - centrifugal_torque
        phi_ddot = (self.g / self.h) * np.sin(self.phi) - (
            self.v**2 / (self.L * self.h)
        ) * np.tan(self.delta) * np.cos(self.phi)
        self.phi_dot += phi_ddot * self.dt
        self.phi += self.phi_dot * self.dt

        # Decay steering towards zero
        self.delta *= 0.95

        # Update position
        self.theta += (self.v / self.L) * np.tan(self.delta) * self.dt
        self.x += self.v * np.cos(self.theta) * self.dt
        self.z += self.v * np.sin(self.theta) * self.dt

        # Termination & Reward
        done = False

        new_dist_to_goal = np.linalg.norm(self.goal_pos - np.array([self.x, self.z]))

        # 1. Reward for making progress towards the goal
        progress_reward = (self.dist_to_goal - new_dist_to_goal) * 10.0
        self.dist_to_goal = new_dist_to_goal

        # 2. Reward for staying upright
        upright_reward = (1.0 - (abs(self.phi) / self.max_phi) ** 0.5) * 0.2

        # 3. Reward for heading towards the goal
        heading_vector = np.array([np.cos(self.theta), np.sin(self.theta)])
        goal_vector = self.goal_pos - np.array([self.x, self.z])
        norm_goal_vector = goal_vector / (
            new_dist_to_goal if new_dist_to_goal > 0 else 1.0
        )
        heading_reward = np.dot(heading_vector, norm_goal_vector) * 0.3

        # 4. Small penalty for steering to encourage straight riding
        steering_penalty = -(abs(self.delta) / self.max_delta) * 0.1

        reward = progress_reward + upright_reward + heading_reward + steering_penalty

        if abs(self.phi) > self.max_phi:
            reward = -10.0
            done = True

        if self.steps > 2000:
            done = True

        if new_dist_to_goal < 2.0:
            reward = 50.0
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        vec_to_goal = self.goal_pos - np.array([self.x, self.z])
        dist = np.linalg.norm(vec_to_goal)
        norm_vec_to_goal = vec_to_goal / dist if dist > 0 else np.zeros(2)

        return np.array(
            [
                self.phi,
                self.phi_dot,
                self.delta,
                np.cos(self.theta),
                np.sin(self.theta),
                norm_vec_to_goal[0],
                norm_vec_to_goal[1],
            ]
        )

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "pos": [self.x, self.z],
            "theta": self.theta,
            "phi": self.phi,
            "delta": self.delta,
            "wheelbase": self.L,
            "goal_pos": self.goal_pos.tolist(),
            "bounds": [60, 60],  # For camera
        }


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_bicycle(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "bicycle")


def infer_action_bicycle(obs: List[float], model_filename: str | None = None) -> int:
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("bicycle", obs, model_filename)


async def run_bicycle(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "bicycle",
        BicycleEnv,
        model_filename=model_filename,
        action_transform=int,
    )
