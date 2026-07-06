from typing import List, Dict, Any

import numpy as np
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# Glider Environment
# -----------------------------------------------------------------------------------


class GliderEnv:
    """A glider environment for learning dynamic soaring."""

    def __init__(self):
        # Physics constants
        self.g = 9.81
        self.mass = 1.5
        self.rho = 1.225  # air density
        self.S = 0.5  # wing area
        self.CL_alpha = 2 * np.pi  # Lift coefficient slope
        self.CD0 = 0.02  # Parasitic drag
        self.CD_k = 0.05  # Induced drag factor
        self.dt = 0.02

        # Waypoints for navigation task
        self.waypoints = [np.array([-160.0, 0.0, 70.0]), np.array([160.0, 0.0, 70.0])]
        self.current_waypoint_index = 0
        self.waypoint_threshold = 15.0  # How close to get to a waypoint

        # Wind model (updrafts for thermal soaring)
        self.wind_C1 = 8.0  # max updraft/downdraft speed (m/s)
        self.wind_C2 = 0.1  # Unused
        self.wind_C3 = 50.0  # Unused
        self.wind_wave_freq = 1.0 / 250.0  # Spatial frequency of thermals
        self.wind_wave_mag = 1.0  # Multiplier for first wave
        self.wind_wave_freq2 = (
            1.0 / 400.0
        )  # Spatial frequency of a second, wider thermal wave
        self.wind_wave_mag2 = 0.7  # Multiplier for second wave

        # State variables
        self.pos = np.zeros(3)  # x, y, z
        self.vel = np.zeros(3)  # vx, vy, vz
        self.rot = np.zeros(3)  # roll, pitch, yaw (phi, theta, psi)
        self.ang_vel = np.zeros(3)  # p, q, r

        self.max_roll = np.pi / 2
        self.max_pitch = np.pi / 4
        self.max_aoa = np.deg2rad(15)

        self.steps = 0
        self.reset()

    def _wind_model(self, pos: np.ndarray) -> np.ndarray:
        x, y = pos[0], pos[1]

        # Create thermal-like vertical currents based on horizontal position (x, y).
        # Superimposing sine waves to create less regular patterns of lift and sink.
        updraft1 = (
            np.sin(x * self.wind_wave_freq * 2 * np.pi)
            * np.cos(y * self.wind_wave_freq * 2 * np.pi)
            * self.wind_C1
            * self.wind_wave_mag
        )

        updraft2 = (
            np.sin(x * self.wind_wave_freq2 * 2 * np.pi / 1.5)
            * np.cos(y * self.wind_wave_freq * 2 * np.pi / 1.5)
            * self.wind_C1
            * self.wind_wave_mag2
        )

        total_updraft = updraft1 + updraft2

        # A very gentle horizontal wind to avoid zero-velocity issues and give a slight drift
        return np.array([1.0, 0.5, total_updraft])

    def reset(self):
        self.pos = np.array([0.0, 0.0, 60.0])
        self.vel = np.array([15.0, 0.0, -1.0])  # Reduced initial velocity
        self.rot = np.zeros(3)
        self.ang_vel = np.random.uniform(-0.1, 0.1, 3)
        self.current_waypoint_index = np.random.randint(0, len(self.waypoints))
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1

        # --- Control inputs ---
        # Actions: 0:none, 1:roll_left, 2:roll_right, 3:pitch_up, 4:pitch_down
        roll_torque, pitch_torque, yaw_torque = 0.0, 0.0, 0.0
        if action == 1:
            roll_torque = -15.0
            yaw_torque = 4.0  # Coordinated turn
        elif action == 2:
            roll_torque = 15.0
            yaw_torque = -4.0  # Coordinated turn
        elif action == 3:
            pitch_torque = 10.0
        elif action == 4:
            pitch_torque = -10.0

        # --- Physics update ---
        # Simplified rotational dynamics
        self.ang_vel[0] += roll_torque * self.dt
        self.ang_vel[1] += pitch_torque * self.dt
        self.ang_vel[2] += yaw_torque * self.dt
        self.ang_vel *= 0.95  # damping
        self.rot += self.ang_vel * self.dt

        # Clamp rotation
        self.rot[0] = np.clip(self.rot[0], -self.max_roll, self.max_roll)
        self.rot[1] = np.clip(self.rot[1], -self.max_pitch, self.max_pitch)

        # Get forces
        wind = self._wind_model(self.pos)
        v_air = self.vel - wind
        v_air_mag = np.linalg.norm(v_air)

        # Simplified aerodynamics
        aoa = np.arctan2(-v_air[2], v_air[0]) if v_air[0] != 0 else 0

        if v_air_mag > 0.1:
            CL = self.CL_alpha * aoa
            CD = self.CD0 + self.CD_k * CL**2
            lift_mag = 0.5 * self.rho * v_air_mag**2 * self.S * CL
            drag_mag = 0.5 * self.rho * v_air_mag**2 * self.S * CD

            # Simplified force vectors in body frame
            lift_force = np.array([0, 0, lift_mag])
            drag_force = np.array([-drag_mag, 0, 0])

            # Rotate forces from body to world frame
            R_roll = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(self.rot[0]), -np.sin(self.rot[0])],
                    [0, np.sin(self.rot[0]), np.cos(self.rot[0])],
                ]
            )
            R_pitch = np.array(
                [
                    [np.cos(self.rot[1]), 0, np.sin(self.rot[1])],
                    [0, 1, 0],
                    [-np.sin(self.rot[1]), 0, np.cos(self.rot[1])],
                ]
            )
            R_yaw = np.array(
                [
                    [np.cos(self.rot[2]), -np.sin(self.rot[2]), 0],
                    [np.sin(self.rot[2]), np.cos(self.rot[2]), 0],
                    [0, 0, 1],
                ]
            )
            R = R_yaw @ R_pitch @ R_roll

            total_aero_force = R @ (lift_force + drag_force)
        else:
            total_aero_force = np.zeros(3)
            aoa = 0

        gravity_force = np.array([0, 0, -self.mass * self.g])
        total_force = total_aero_force + gravity_force

        # Integration
        self.vel += (total_force / self.mass) * self.dt
        self.pos += self.vel * self.dt

        # --- Termination & Reward ---
        done = False

        # Waypoint navigation logic
        target_waypoint = self.waypoints[self.current_waypoint_index]
        vec_to_target = target_waypoint - self.pos
        dist_to_target = np.linalg.norm(vec_to_target)

        if dist_to_target < self.waypoint_threshold:
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(
                self.waypoints
            )

        # Reward based on heading and energy
        vel_dir = self.vel / (np.linalg.norm(self.vel) + 1e-8)
        target_dir = vec_to_target / (dist_to_target + 1e-8)
        heading_alignment = np.dot(vel_dir, target_dir)

        # Scale to [0, 1]
        H = (heading_alignment + 1) / 2

        v_mag = np.linalg.norm(self.vel)
        # Scale velocity to a reasonable range for energy metric
        E = np.clip(v_mag / 30.0, 0, 2.0)

        # Paper's mixed reward: R = E * (H - E + 1)
        # This incentivizes high energy when heading is good, and energy seeking when it's low
        reward = E * (H - E + 1)

        # --- Penalties to constrain behavior ---

        # Penalty for straying from the central corridor (y-axis deviation)
        lateral_dist = abs(self.pos[1])
        corridor_half_width = 250.0
        if lateral_dist > corridor_half_width:
            # Quadratic penalty that grows stronger the further it deviates
            penalty_ratio = (lateral_dist - corridor_half_width) / 100.0
            reward -= 2.0 * (penalty_ratio**2)

        # Penalty for altitude deviations
        altitude = self.pos[2]
        if altitude > 250.0:
            # Quadratic penalty for being too high
            reward -= 2.0 * ((altitude - 250.0) / 50.0) ** 2
        elif altitude < 25.0:
            # Small penalty for being low, encouraging it to stay up
            reward -= 0.5

        # Penalty for crashing
        if self.pos[2] < 5.0:
            reward = -50.0
            done = True

        # Penalty for stalling
        if abs(aoa) > self.max_aoa:
            reward = -50.0
            done = True

        # Penalty for flying too far away from the action
        if dist_to_target > 500:  # If it's more than 1.5x the waypoint separation
            reward = -50.0
            done = True

        # Timeout
        if self.steps > 4000:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        # Observations for the agent
        target_waypoint = self.waypoints[self.current_waypoint_index]
        vec_to_target = target_waypoint - self.pos
        dist_to_target = np.linalg.norm(vec_to_target)
        dir_to_target = vec_to_target / (dist_to_target + 1e-8)

        return np.concatenate(
            [
                np.array(
                    [
                        self.vel[2] / 10.0,  # vertical speed (normalized)
                        (self.pos[2] - self.wind_C3)
                        / 50.0,  # height relative to wind layer (normalized)
                        self.rot[0],  # roll
                        self.rot[1],  # pitch
                        np.sin(self.rot[2]),  # yaw sin
                        np.cos(self.rot[2]),  # yaw cos
                        self.ang_vel[0],  # roll rate
                        self.ang_vel[1],  # pitch rate
                        self.ang_vel[2],  # yaw rate
                    ]
                ),
                self.vel / 20.0,  # velocity (normalized)
                dir_to_target,  # direction to target
                [dist_to_target / 100.0],  # distance to target (normalized)
            ]
        )

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "pos": self.pos.tolist(),
            "rot": self.rot.tolist(),  # roll, pitch, yaw
            "wind_params": [
                self.wind_C1,
                self.wind_C2,
                self.wind_C3,
                self.wind_wave_freq,
                self.wind_wave_mag,
                self.wind_wave_freq2,
                self.wind_wave_mag2,
            ],
            "bounds": [400, 400],
            "waypoints": [w.tolist() for w in self.waypoints],
            "current_waypoint_index": self.current_waypoint_index,
        }


# -----------------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# -----------------------------------------------------------------------------------


async def train_glider(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(websocket, "glider")


def infer_action_glider(obs: List[float], model_filename: str | None = None) -> int:
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action("glider", obs, model_filename)


async def run_glider(websocket: WebSocket, model_filename: str | None = None):
    from mlagents.websocket_training import run_policy_for_websocket

    await run_policy_for_websocket(
        websocket,
        "glider",
        GliderEnv,
        model_filename=model_filename,
        action_transform=int,
    )
