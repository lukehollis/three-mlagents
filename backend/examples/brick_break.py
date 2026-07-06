from typing import List, Dict, Any

import numpy as np
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# BrickBreak Environment
# -----------------------------------------------------------------------------------


class BrickBreakEnv:
    """A simple BrickBreak game environment."""

    def __init__(
        self,
        width=40,
        height=40,
        paddle_width=8,
        ball_radius=1,
        brick_rows=5,
        brick_cols=8,
    ):
        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.ball_radius = ball_radius
        self.brick_rows = brick_rows
        self.brick_cols = brick_cols
        self.brick_width = width / brick_cols
        self.brick_height = 2

        self.paddle_x = 0
        self.ball_pos = np.zeros(2)
        self.ball_vel = np.zeros(2)
        self.bricks = np.ones((brick_rows, brick_cols))

        self.reset()

    def reset(self):
        self.paddle_x = self.width / 2
        self.ball_pos = np.array([self.width / 2, self.height / 4])
        angle = np.random.uniform(np.pi / 4, 3 * np.pi / 4)
        self.ball_vel = np.array([np.cos(angle), np.sin(angle)]) * 1.5
        self.bricks = np.ones((self.brick_rows, self.brick_cols))
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1
        # Action: 0=left, 1=stay, 2=right
        if action == 0:
            self.paddle_x -= 3
        elif action == 2:
            self.paddle_x += 3

        self.paddle_x = np.clip(
            self.paddle_x, self.paddle_width / 2, self.width - self.paddle_width / 2
        )

        # Update ball position
        self.ball_pos += self.ball_vel

        # Collisions
        reward = 0.0

        # Wall collisions
        if (
            self.ball_pos[0] <= self.ball_radius
            or self.ball_pos[0] >= self.width - self.ball_radius
        ):
            self.ball_vel[0] *= -1
        if self.ball_pos[1] >= self.height - self.ball_radius:
            self.ball_vel[1] *= -1

        # Paddle collision
        if (
            self.ball_vel[1] < 0
            and self.ball_pos[1] - self.ball_radius <= 2
            and self.ball_pos[0] >= self.paddle_x - self.paddle_width / 2
            and self.ball_pos[0] <= self.paddle_x + self.paddle_width / 2
        ):
            self.ball_vel[1] *= -1
            offset = (self.ball_pos[0] - self.paddle_x) / (self.paddle_width / 2)
            self.ball_vel[0] += offset * 0.5
            reward = 0.1

        # Brick collisions
        brick_y_start = self.height - self.brick_rows * self.brick_height - 10
        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if self.bricks[r, c] == 1:
                    brick_x = c * self.brick_width
                    brick_y = brick_y_start + r * self.brick_height
                    if (
                        self.ball_pos[0] >= brick_x
                        and self.ball_pos[0] <= brick_x + self.brick_width
                        and self.ball_pos[1] >= brick_y
                        and self.ball_pos[1] <= brick_y + self.brick_height
                    ):
                        self.bricks[r, c] = 0
                        self.ball_vel[1] *= -1
                        reward = 1.0
                        break
            if reward == 1.0:
                break

        # Termination
        done = False
        if self.ball_pos[1] < self.ball_radius:
            reward = -1.0
            done = True

        if np.sum(self.bricks) == 0:
            reward = 10.0
            done = True

        if self.steps > 2000:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        return np.concatenate(
            [
                self.ball_pos / np.array([self.width, self.height]),
                self.ball_vel,
                [self.paddle_x / self.width],
                self.bricks.flatten(),
            ]
        )

    def get_state_for_viz(self) -> Dict[str, Any]:
        brick_list = []
        brick_y_start = self.height - self.brick_rows * self.brick_height - 10
        for r in range(self.brick_rows):
            for c in range(self.brick_cols):
                if self.bricks[r, c] == 1:
                    brick_list.append(
                        {
                            "pos": [
                                c * self.brick_width + self.brick_width / 2,
                                brick_y_start
                                + r * self.brick_height
                                + self.brick_height / 2,
                            ],
                            "size": [self.brick_width * 0.9, self.brick_height * 0.8],
                        }
                    )
        return {
            "ball": {"pos": self.ball_pos.tolist(), "radius": self.ball_radius},
            "paddle": {"pos": [self.paddle_x, 1], "size": [self.paddle_width, 2]},
            "bricks": brick_list,
            "bounds": [self.width, self.height],
        }


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_brick_break(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "brickbreak")


def infer_action_brick_break(
    obs: List[float], model_filename: str | None = None
) -> int:
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("brickbreak", obs, model_filename)


async def run_brick_break(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "brickbreak",
        BrickBreakEnv,
        model_filename=model_filename,
        action_transform=int,
    )
