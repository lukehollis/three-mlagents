import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any
from collections import deque

import asyncio
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime
from fastapi import WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


ACTION_SIZE = 7  # 0=no thrust, 1-6=thrust directions
POLICIES_DIR = "policies"
# BATCH_SIZE = 8192 - Handled by SB3 n_steps
# MINI_BATCH = 512 - Handled by SB3 batch_size
# EPOCHS = 10 - Handled by SB3 n_epochs
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 1e-4
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 120000  # Increased episode length
TOTAL_TIMESTEPS = MAX_STEPS_PER_EPISODE * EPISODES # Corresponds to new timeout logic

# -----------------------------------------------------------------------------------
# Astrodynamics Environment
# -----------------------------------------------------------------------------------

class AstrodynamicsEnv(gym.Env):
    """An orbital rendezvous environment for spacecraft navigation."""
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, training_mode: bool = True):
        super().__init__()
        # Physical constants
        self.mu = 3.986e14  # Earth's gravitational parameter (m³/s²)
        self.earth_radius = 6.371e6  # Earth radius (m)
        
        # Target orbit (MEO - Medium Earth Orbit)
        self.orbit_altitude = 15000e3  # Target orbital altitude (m)
        self.orbit_radius = self.earth_radius + self.orbit_altitude
        self.orbital_velocity = np.sqrt(self.mu / self.orbit_radius)
        
        # Spacecraft starting orbit (LEO - Low Earth Orbit)
        self.leo_altitude = 400e3 # 400 km altitude
        self.leo_radius = self.earth_radius + self.leo_altitude
        self.leo_velocity = np.sqrt(self.mu / self.leo_radius)

        # Spacecraft parameters
        self.mass = 1000.0  # kg
        self.max_thrust = 500000.0  # N (Maneuvering thrusters for the agent)
        self.specific_impulse = 300000.0  # seconds
        self.initial_fuel = 500000.0  # kg (enough for a long burn)
        self.dt = 10.0  # seconds (smaller timestep for stability)
        
        # Mission parameters
        self.docking_threshold = 50.0  # meters
        self.velocity_threshold = 2.0  # m/s for successful docking
        self.max_distance = 100e6  # maximum distance from target (m)
        self.docking_approach_threshold = 10000.0 # 10 km, for reward shaping
        
        # State variables
        # Relative state (in orbital reference frame of the target)
        self.relative_pos = np.zeros(3)  # relative position to target (m)
        self.relative_vel = np.zeros(3)  # relative velocity to target (m/s)
        
        # Absolute state (in Earth-Centered Inertial frame)
        self.spacecraft_pos_abs = np.zeros(3)
        self.spacecraft_vel_abs = np.zeros(3)
        self.target_pos_abs = np.zeros(3)
        self.target_vel_abs = np.zeros(3)
        
        self.fuel = self.initial_fuel  # remaining fuel (kg)
        self.target_pos = np.zeros(3)  # target space station position (in relative frame, so always 0)
        self.target_vel = np.zeros(3)  # target space station velocity (in relative frame, so always 0)
        
        # Visualization trail
        self.max_trail_length = 10000
        self.spacecraft_trail = deque(maxlen=self.max_trail_length)
        self.target_trail = deque(maxlen=self.max_trail_length)
        
        self.steps = 0
        self.training_mode = training_mode
        
        # Episode tracking for callback
        self.total_episode_reward = 0.0
        
        self.action_space = spaces.Discrete(ACTION_SIZE)
        obs_shape = self._get_obs_dummy().shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    def _get_derivatives(self, state: np.ndarray, thrust_vector: np.ndarray, current_mass: float) -> np.ndarray:
        """
        Calculates the derivatives of the state vector (d_pos/dt, d_vel/dt)
        for use in the RK4 integrator.
        """
        pos = state[0:3]
        vel = state[3:6]
        
        grav_accel = self._orbital_dynamics(pos)
        
        thrust_accel = np.zeros(3)
        if self.fuel > 0 and current_mass > 0:
            thrust_accel = thrust_vector / current_mass
            
        total_accel = grav_accel + thrust_accel
        
        return np.concatenate([vel, total_accel])

    def _orbital_dynamics(self, pos_global: np.ndarray) -> np.ndarray:
        """Calculate gravitational acceleration at given global position."""
        r = np.linalg.norm(pos_global)
        if r < self.earth_radius:
            r = self.earth_radius  # Prevent division by zero
        
        # Gravitational acceleration vector pointing toward Earth center
        accel = -self.mu * pos_global / (r**3)
        return accel

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Target starts in a circular MEO in the XY plane
        self.target_pos_abs = np.array([self.orbit_radius, 0.0, 0.0])
        self.target_vel_abs = np.array([0.0, self.orbital_velocity, 0.0])

        # Randomly choose a starting scenario between LEO and a higher orbit.
        start_scenario = np.random.choice(['leo', 'outer_orbit'])

        if start_scenario == 'leo':
            # Start spacecraft in a random position in LEO in the same plane
            start_angle = np.random.uniform(0, 2 * np.pi)
            self.spacecraft_pos_abs = np.array([
                self.leo_radius * np.cos(start_angle),
                self.leo_radius * np.sin(start_angle),
                0.0
            ])
            
            # Velocity is tangential to the orbit
            self.spacecraft_vel_abs = np.array([
                -self.leo_velocity * np.sin(start_angle),
                self.leo_velocity * np.cos(start_angle),
                0.0
            ])
        
        elif start_scenario == 'outer_orbit':
            # Start in a random higher orbit, further away than the target
            outer_radius = np.random.uniform(self.orbit_radius * 1.2, self.orbit_radius * 2.5)
            outer_velocity = np.sqrt(self.mu / outer_radius)
            start_angle = np.random.uniform(0, 2 * np.pi)
            self.spacecraft_pos_abs = np.array([
                outer_radius * np.cos(start_angle),
                outer_radius * np.sin(start_angle),
                0.0
            ])
            self.spacecraft_vel_abs = np.array([
                -outer_velocity * np.sin(start_angle),
                outer_velocity * np.cos(start_angle),
                0.0
            ])
        
        # Calculate initial relative state
        self.relative_pos = self.spacecraft_pos_abs - self.target_pos_abs
        self.relative_vel = self.spacecraft_vel_abs - self.target_vel_abs
        
        self.fuel = self.initial_fuel
        self.spacecraft_trail.clear()
        self.target_trail.clear()
        self.steps = 0
        
        # Reset episode tracking
        self.total_episode_reward = 0.0
        
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps += 1

        # --- Define local reference frame (Up, North, East) ---
        up_dir = self.spacecraft_pos_abs / (np.linalg.norm(self.spacecraft_pos_abs) + 1e-8)
        
        # North direction (aligns with Earth's rotational axis)
        # To get a consistent North, project world Z onto the tangent plane
        z_axis = np.array([0, 0, 1])
        north_dir = z_axis - np.dot(z_axis, up_dir) * up_dir
        north_dir /= (np.linalg.norm(north_dir) + 1e-8)

        # East direction is orthogonal to North and Up
        east_dir = np.cross(north_dir, up_dir)

        # Action mapping: 0=no thrust, 1-6=thrust in local frame
        thrust_vector = np.zeros(3)
        thrust_magnitude = self.max_thrust
        
        local_thrust_direction = np.zeros(3)
        if action == 1:   # +Up (main thrust)
            local_thrust_direction = np.array([1, 0, 0])
        elif action == 2: # -Up (retro/down)
            local_thrust_direction = np.array([-1, 0, 0])
        elif action == 3: # +North
            local_thrust_direction = np.array([0, 1, 0])
        elif action == 4: # -North
            local_thrust_direction = np.array([0, -1, 0])
        elif action == 5: # +East
            local_thrust_direction = np.array([0, 0, 1])
        elif action == 6: # -East
            local_thrust_direction = np.array([0, 0, -1])
        
        if action > 0:
            # Convert local thrust direction to global ECI frame
            # The basis vectors (up, north, east) form the columns of the transformation matrix
            rotation_matrix = np.column_stack([up_dir, north_dir, east_dir])
            global_thrust_dir = rotation_matrix @ local_thrust_direction
            thrust_vector = global_thrust_dir * thrust_magnitude
        
        # Calculate fuel consumption
        thrust_mag = np.linalg.norm(thrust_vector)
        if thrust_mag > 0:
            fuel_consumed = thrust_mag * self.dt / (self.specific_impulse * 9.81)
            self.fuel = max(0, self.fuel - fuel_consumed)
            
            # Reduce thrust if out of fuel
            if self.fuel <= 0:
                thrust_vector = np.zeros(3)

        # --- Integrate motion using Runge-Kutta 4 (RK4) for accuracy ---
        current_mass = self.mass + self.fuel

        # State vector for spacecraft
        state_sc = np.concatenate([self.spacecraft_pos_abs, self.spacecraft_vel_abs])

        # RK4 integration for spacecraft
        k1_sc = self._get_derivatives(state_sc, thrust_vector, current_mass)
        k2_sc = self._get_derivatives(state_sc + 0.5 * self.dt * k1_sc, thrust_vector, current_mass)
        k3_sc = self._get_derivatives(state_sc + 0.5 * self.dt * k2_sc, thrust_vector, current_mass)
        k4_sc = self._get_derivatives(state_sc + self.dt * k3_sc, thrust_vector, current_mass)
        self.spacecraft_pos_abs += (self.dt / 6.0) * (k1_sc[0:3] + 2*k2_sc[0:3] + 2*k3_sc[0:3] + k4_sc[0:3])
        self.spacecraft_vel_abs += (self.dt / 6.0) * (k1_sc[3:6] + 2*k2_sc[3:6] + 2*k3_sc[3:6] + k4_sc[3:6])

        # State vector for target (zero thrust and mass)
        state_t = np.concatenate([self.target_pos_abs, self.target_vel_abs])
        
        # RK4 integration for target
        k1_t = self._get_derivatives(state_t, np.zeros(3), 0)
        k2_t = self._get_derivatives(state_t + 0.5 * self.dt * k1_t, np.zeros(3), 0)
        k3_t = self._get_derivatives(state_t + 0.5 * self.dt * k2_t, np.zeros(3), 0)
        k4_t = self._get_derivatives(state_t + self.dt * k3_t, np.zeros(3), 0)
        self.target_pos_abs += (self.dt / 6.0) * (k1_t[0:3] + 2*k2_t[0:3] + 2*k3_t[0:3] + k4_t[0:3])
        self.target_vel_abs += (self.dt / 6.0) * (k1_t[3:6] + 2*k2_t[3:6] + 2*k3_t[3:6] + k4_t[3:6])
        
        # Update relative state for observations and rewards
        self.relative_pos = self.spacecraft_pos_abs - self.target_pos_abs
        self.relative_vel = self.spacecraft_vel_abs - self.target_vel_abs

        # Update trail for visualization
        self.spacecraft_trail.append(self.spacecraft_pos_abs.copy())
        self.target_trail.append(self.target_pos_abs.copy())

        # Calculate reward and termination
        distance = np.linalg.norm(self.relative_pos)
        velocity_mag = np.linalg.norm(self.relative_vel)
        spacecraft_radius = np.linalg.norm(self.spacecraft_pos_abs)

        # First, check for terminal conditions which apply in all phases
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if not self.training_mode:
            if spacecraft_radius < self.earth_radius:
                terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        # Universal Failure & Success Conditions
        episode_end_reason = None
        if spacecraft_radius < self.earth_radius:
            print(f"Episode End: Crashed into Earth at step {self.steps}.")
            reward = -200.0; terminated = True
            episode_end_reason = "Crashed into Earth"
        elif distance > self.max_distance:
            print(f"Episode End: Exceeded max distance at step {self.steps}.")
            reward = -10.0; terminated = True
            episode_end_reason = "Exceeded max distance"
        elif self.fuel <= 0 and distance > self.docking_threshold:
            print(f"Episode End: Out of fuel at step {self.steps}.")
            reward = -50.0; terminated = True
            episode_end_reason = "Out of fuel"
        elif distance < self.docking_threshold and velocity_mag > self.velocity_threshold:
            print(f"Episode End: Crashed into target at step {self.steps}.")
            reward = -50.0; terminated = True
            episode_end_reason = "Crashed into target"
        elif self.steps > MAX_STEPS_PER_EPISODE:
            print(f"Episode End: Timeout at step {self.steps}.")
            reward = -5.0; truncated = True
            episode_end_reason = "Timeout"
        elif distance < self.docking_threshold and velocity_mag < self.velocity_threshold:
            print(f"Episode End: Successfully docked at step {self.steps}.")
            reward = 1000.0; terminated = True
            episode_end_reason = "Successfully docked"
        
        # Track total episode reward
        self.total_episode_reward += reward
        
        if terminated or truncated:
            # Add episode end to queue for callback
            if episode_end_reason and self.training_mode:
                info["episode_end"] = {
                    "reason": episode_end_reason,
                    "steps": self.steps,
                    "total_reward": self.total_episode_reward,
                }
            return self._get_obs(), reward, terminated, truncated, info

        # --- State-Dependent Reward Logic ---
        # The agent ALWAYS starts in a stable orbit. Its goal is to maneuver to the target.
        if distance > self.docking_approach_threshold:
            # PHASE 1: ORBITAL RENDEZVOUS (replaces old Phase 1 & 2).
            # Guide the agent to match the target's orbital parameters.
            target_radius = self.orbit_radius
            altitude_scale = self.orbit_altitude - self.leo_altitude
            up_dir = self.spacecraft_pos_abs / (spacecraft_radius + 1e-8)

            # 1. Altitude Reward: Gaussian peak at the target orbit.
            radius_diff = spacecraft_radius - target_radius
            radius_reward = np.exp(-(radius_diff / (altitude_scale * 0.1))**2) * 50.0
            reward += radius_reward
            
            # 2. Orbital Velocity Reward
            radial_velocity_component = np.dot(self.spacecraft_vel_abs, up_dir) * up_dir
            tangential_velocity_vec = self.spacecraft_vel_abs - radial_velocity_component
            tangential_velocity_mag = np.linalg.norm(tangential_velocity_vec)
            velocity_diff = tangential_velocity_mag - self.orbital_velocity
            velocity_match_reward = np.exp(-(velocity_diff / (self.orbital_velocity * 0.15))**2) * 40.0
            altitude_proximity = np.exp(-(radius_diff / (altitude_scale * 0.5))**2)
            reward += altitude_proximity * velocity_match_reward

            # 3. Approach Reward - log-scaled penalty for distance
            distance_penalty = np.log1p(distance / 1000.0) * 0.5 # distance in km
            reward -= distance_penalty

            # 4. Orbital Energy Reward
            target_energy = -self.mu / (2 * self.orbit_radius)
            current_radius = np.linalg.norm(self.spacecraft_pos_abs)
            current_speed = np.linalg.norm(self.spacecraft_vel_abs)
            if current_radius < 1.0: current_radius = 1.0 # Avoid division by zero
            current_energy = (current_speed**2 / 2) - (self.mu / current_radius)
            energy_diff = np.abs(current_energy - target_energy)
            energy_match_reward = np.exp(-(energy_diff / np.abs(target_energy)) * 2.0) * 35.0
            reward += energy_match_reward
        else:
            # PHASE 2: DOCKING (Terminal Guidance)
            # Once close, focus on killing relative velocity and position.
            reward_gate = (1.0 - (distance / self.docking_approach_threshold))

            # 1. Reward for distance reduction
            distance_reward = (1.0 - (distance / self.docking_approach_threshold)) * 25.0
            reward += distance_reward

            # 2. Velocity matching reward (critical at close range)
            velocity_reward = np.exp(-(velocity_mag / self.velocity_threshold)**2) * 50.0
            reward += reward_gate * velocity_reward

            # 3. Fuel efficiency reward when close
            if action == 0:
                reward += reward_gate * 0.5

        # Small universal time penalty
        reward -= 0.1
        
        # Track total episode reward for non-terminal steps too
        self.total_episode_reward += reward
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs_dummy(self):
        """Helper to get dummy observation for space definition."""
        self.relative_pos = np.zeros(3)
        self.relative_vel = np.zeros(3)
        self.fuel = self.initial_fuel
        self.steps = 0
        return self._get_obs()
        
    def _get_obs(self):
        """Get observation vector for the agent."""
        distance = np.linalg.norm(self.relative_pos)
        velocity_mag = np.linalg.norm(self.relative_vel)
        
        # Normalize observations
        pos_norm = self.relative_pos / self.max_distance
        vel_norm = self.relative_vel / 10000.0     # Normalize by a typical orbital velocity delta
        
        # Unit vector pointing to target
        target_dir = -self.relative_pos / (distance + 1e-8)
        
        return np.concatenate([
            pos_norm,                          # Relative position (3)
            vel_norm,                          # Relative velocity (3)
            target_dir,                        # Direction to target (3)
            [distance / self.max_distance],    # Distance to target (1)
            [velocity_mag / 10000.0],          # Speed magnitude (1)
            [self.fuel / self.initial_fuel],   # Fuel ratio (1)
            [self.steps / MAX_STEPS_PER_EPISODE]             # Time progress (1)
        ])

    def get_state_for_viz(self) -> Dict[str, Any]:
        """Get state information for visualization."""
        return {
            "spacecraft_pos": self.relative_pos.tolist(),
            "spacecraft_vel": self.relative_vel.tolist(),
            "spacecraft_pos_abs": self.spacecraft_pos_abs.tolist(),
            "spacecraft_vel_abs": self.spacecraft_vel_abs.tolist(),
            "target_pos_abs": self.target_pos_abs.tolist(),
            "target_pos": self.target_pos.tolist(),
            "fuel_ratio": self.fuel / self.initial_fuel,
            "distance_to_target": float(np.linalg.norm(self.relative_pos)),
            "velocity_magnitude": float(np.linalg.norm(self.relative_vel)),
            "trail": [pos.tolist() for pos in self.spacecraft_trail],
            "target_trail": [pos.tolist() for pos in self.target_trail],
            "orbit_params": {
                "radius": self.orbit_radius,
                "velocity": self.orbital_velocity,
                "leo_radius": self.leo_radius
            }
        }


# -----------------------------------------------------------------------------------
# Callbacks and training setup
# -----------------------------------------------------------------------------------

class WebSocketCallback(BaseCallback):
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop, verbose=0):
        super(WebSocketCallback, self).__init__(verbose)
        self.websocket = websocket
        self.loop = loop
        self.should_stop = False
        self.episode_count = 0
        self.last_timesteps = 0
        self.failed_sends = 0
        self.max_failed_sends = 1000
        # Visualization tracking
        self.vis_env_idx = 0 # The environment to visualize
        self.step_counter = 0 # Counter for cycling through environments

    def _send_message_safely(self, payload: dict) -> bool:
        """Send a message via WebSocket with proper error handling."""
        if self.should_stop or self.websocket.application_state != WebSocketState.CONNECTED:
            if not self.should_stop:
                print("WebSocket is not connected. Stopping training.")
                self.should_stop = True
            return False

        try:
            future = asyncio.run_coroutine_threadsafe(
                send_json_safely(self.websocket, payload),
                self.loop
            )
            future.result(timeout=1.0)  # Increased timeout for more reliability
            return True
        except (WebSocketDisconnect, ConnectionClosedError):
            print("Stopping training due to client disconnect.")
            self.should_stop = True
            return False
        except Exception as e:
            self.failed_sends += 1
            if self.failed_sends % 100 == 0:  # Report errors less frequently
                print(f"Error sending WebSocket message (attempt {self.failed_sends}): {e}")
            
            if self.failed_sends >= self.max_failed_sends:
                print(f"Too many failed WebSocket sends ({self.failed_sends}). Stopping training.")
                self.should_stop = True
            return False

    def _on_step(self) -> bool:
        if self.should_stop:
            return False

        self.step_counter += 1
        
        # Cycle through environments to ensure continuous updates
        self.vis_env_idx = self.step_counter % self.training_env.num_envs

        # Check for episode ends from all environments.
        # This is the standard SB3 way to get info from completed episodes.
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "episode_end" in info:
                    self.episode_count += 1
                    episode_info = info["episode_end"]
                    
                    # --- Send final state of the episode that just ended ---
                    try:
                        states = self.training_env.env_method("get_state_for_viz", indices=[i])
                        if states and states[0]:
                            final_state_payload = {
                                "type": "train_step",
                                "state": states[0],
                                "episode": self.episode_count,
                                "timestep": self.num_timesteps,
                            }
                            if not self._send_message_safely(final_state_payload):
                                return False
                    except Exception as e:
                        print(f"Error getting final viz state for env {i}: {e}")
                    # ---------------------------------------------------------

                    print(
                        f"Episode {self.episode_count} (env {i}): {episode_info['reason']} at step {episode_info['steps']} "
                        f"(reward: {episode_info['total_reward']:.2f})"
                    )

                    payload = {
                        "type": "episode_end",
                        "episode": self.episode_count,
                        "timestep": self.num_timesteps,
                        "reason": episode_info["reason"],
                        "steps": episode_info["steps"],
                        "reward": episode_info["total_reward"],
                        "env_idx": i,
                    }
                    if not self._send_message_safely(payload):
                        return False
        
        # SEND EVERY STEP TO THE FRONTEND - JUST LIKE THE GLIDER
        try:
            states = self.training_env.env_method("get_state_for_viz", indices=[self.vis_env_idx])
            if states and states[0]:
                state = states[0]
                payload = {
                    "type": "train_step", 
                    "state": state, 
                    "episode": self.episode_count,
                    "timestep": self.num_timesteps
                }
                
                if not self._send_message_safely(payload):
                    return False
                    
        except Exception as e:
            print(f"Error getting environment state for env {self.vis_env_idx}: {e}")
            # Don't stop training for environment state errors
                
        return not self.should_stop

    def _on_rollout_end(self) -> None:
        """Send progress updates after each rollout."""
        if self.should_stop:
            return
            
        # Simple episode counting: estimate based on timesteps and typical episode length
        timesteps_this_rollout = self.num_timesteps - self.last_timesteps
        self.last_timesteps = self.num_timesteps
            
        reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
        loss = self.logger.name_to_value.get("train/loss")

        payload = {
            "type": "progress",
            "episode": self.episode_count,
            "timestep": self.num_timesteps,
            "reward": float(reward) if reward is not None else None,
            "loss": float(loss) if loss is not None else None,
        }
        
        if not self._send_message_safely(payload):
            print("Stopping training due to WebSocket communication failure")

    def stop_training(self):
        """Signal that training should stop."""
        print("Training stop requested via callback")
        self.should_stop = True


def _export_model_onnx(model: PPO, path: str):
    class ExportableModel(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            # Replicate the forward pass of the policy to get logits
            features = self.policy.extract_features(obs)
            latent_pi, _ = self.policy.mlp_extractor(features)
            return self.policy.action_net(latent_pi)

    exportable_model = ExportableModel(model.policy)
    exportable_model.eval()

    dummy_input = torch.randn(1, *model.observation_space.shape)

    torch.onnx.export(
        exportable_model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

# -----------------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------------

# Global variable to track training state
_training_task = None
_training_callback = None


async def send_json_safely(websocket: WebSocket, payload: dict):
    """Safely send JSON over a websocket, ignoring connection errors."""
    if websocket.application_state != WebSocketState.CONNECTED:
        print("Skipping send, websocket not connected.")
        return
    try:
        await websocket.send_json(payload)
    except (WebSocketDisconnect, ConnectionClosedError) as e:
        print(f"Failed to send message: client disconnected. Reason: {e}")
    except Exception as e:
        # This can happen if the event loop is shutting down
        print(f"An unexpected error occurred while sending message: {e}")


async def train_astrodynamics(websocket: WebSocket):
    global _training_task, _training_callback
    
    # Clean up any existing training
    if _training_task and not _training_task.done():
        print("Stopping existing training...")
        if _training_callback:
            _training_callback.stop_training()
        _training_task.cancel()
        try:
            await _training_task
        except asyncio.CancelledError:
            pass
    
    print("Starting astrodynamics training...")
    print(f"WebSocket state: {websocket.application_state}")
    
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs("astrodynamics_tensorboard", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename_base = f"astrodynamics_policy_{ts}_{session_uuid}"
    model_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.zip")
    onnx_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.onnx")
    
    loop = asyncio.get_running_loop()

    def train_model():
        """Synchronous training function to run in executor."""
        try:
            print("Setting up vectorized environment...")
            # Use a vectorized environment for parallel training
            vec_env = make_vec_env(AstrodynamicsEnv, n_envs=32, env_kwargs=dict(training_mode=True))

            policy_kwargs = dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
            )

            print("Creating PPO model...")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                clip_range=CLIP_EPS,
                ent_coef=ENT_COEF,
                learning_rate=LR,
                n_epochs=10,
                batch_size=512,
                n_steps=2048,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"./astrodynamics_tensorboard/"
            )

            print("Setting up WebSocket callback...")
            websocket_callback = WebSocketCallback(websocket, loop)
            global _training_callback
            _training_callback = websocket_callback
            
            print("Starting training loop...")
            # Send initial status to frontend
            initial_payload = {
                "type": "training_started",
                "message": "Training initialized successfully",
                "total_timesteps": TOTAL_TIMESTEPS
            }
            websocket_callback._send_message_safely(initial_payload)
            
            # Manually implement the training loop to be more responsive to cancellation.
            total_timesteps, callback = model._setup_learn(
                TOTAL_TIMESTEPS,
                websocket_callback,
                reset_num_timesteps=True,
                tb_log_name="PPO",
                progress_bar=True # Enable progress bar setup on callback
            )
            
            # The progress bar needs the total_timesteps in locals.
            # We call on_training_start here to initialize callbacks, especially the progress bar.
            callback.on_training_start(
                locals_={"self": model, "total_timesteps": total_timesteps}, 
                globals_=globals()
            )

            print(f"Training for {total_timesteps} timesteps...")
            iteration = 0
            while model.num_timesteps < total_timesteps:
                iteration += 1
                
                if websocket_callback.should_stop:
                    print(f"Training stop signaled at iteration {iteration}. Exiting loop.")
                    break

                print(f"Starting rollout collection iteration {iteration}, timesteps: {model.num_timesteps}/{total_timesteps}")
                
                # The collect_rollouts method will call our callback's _on_step
                continue_training = model.collect_rollouts(
                    model.env,
                    callback=callback,
                    rollout_buffer=model.rollout_buffer,
                    n_rollout_steps=model.n_steps
                )

                if not continue_training:
                    print("Callback requested stop during rollout collection.")
                    break
                
                # Check again before the blocking policy update
                if websocket_callback.should_stop:
                    print("Training stop signaled before policy update.")
                    break
                
                print(f"Training policy update iteration {iteration}...")
                model.train()
                print(f"Completed iteration {iteration}")

            callback.on_training_end()
            print("Training loop completed")

            if not websocket_callback.should_stop:
                # Only save if training completed successfully
                print("Saving model...")
                model.save(model_path)
                _export_model_onnx(model, onnx_path)
                print(f"Model saved to {model_path}")
                return {
                    "success": True,
                    "model_filename_base": model_filename_base,
                    "ts": ts,
                    "session_uuid": session_uuid
                }
            else:
                print("Training stopped early - not saving model")
                return {"success": False, "reason": "Training stopped early"}
                
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            if _training_callback:
                _training_callback.stop_training()
            return {"success": False, "reason": str(e)}
    
    # Run training in executor with proper task management
    print("Starting training task...")
    _training_task = asyncio.create_task(asyncio.to_thread(train_model))
    
    try:
        result = await _training_task
        print(f"Training completed with result: {result}")
        
        if result["success"]:
            print("Sending training completion message...")
            await send_json_safely(websocket, {
                "type": "trained",
                "file_url": f"/policies/{result['model_filename_base']}.zip",
                "model_filename": f"{result['model_filename_base']}.zip",
                "onnx_filename": f"{result['model_filename_base']}.onnx",
                "timestamp": result["ts"],
                "session_uuid": result["session_uuid"]
            })
            print("Training completion message sent")
        else:
            print(f"Training failed: {result.get('reason', 'Unknown error')}")
            await send_json_safely(websocket, {
                "type": "training_error",
                "message": f"Training failed: {result.get('reason', 'Unknown error')}"
            })
            
    except asyncio.CancelledError:
        print("Training task was cancelled. Signaling stop to the training thread.")
        if _training_callback:
            _training_callback.stop_training()
        await send_json_safely(websocket, {
            "type": "training_error", 
            "message": "Training was cancelled by the server."
        })
        # Wait for the thread to finish, but with a timeout
        try:
            await asyncio.wait_for(_training_task, timeout=10.0)
        except asyncio.TimeoutError:
            print("Warning: Training thread did not stop within 10 seconds.")
        except asyncio.CancelledError:
            pass # Task is already cancelled.
        raise
    except Exception as e:
        print(f"Training failed with an unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        if _training_callback:
            _training_callback.stop_training()
        await send_json_safely(websocket, {
            "type": "training_error",
            "message": f"Training failed: {str(e)}"
        })
    finally:
        print("Cleaning up training task...")
        _training_task = None
        _training_callback = None


async def run_simulation(websocket: WebSocket):
    """Runs a physics-only simulation and streams state."""
    env = AstrodynamicsEnv(training_mode=False)
    env.reset()
    from starlette.websockets import WebSocketState, WebSocketDisconnect

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            # Action 0 is no thrust, simulating orbital mechanics
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            state = env.get_state_for_viz()
            await send_json_safely(websocket, {"type": "state", "state": state})
            await asyncio.sleep(0.05)  # ~20Hz update rate
            
            if done:
                env.reset()
    except WebSocketDisconnect:
        print("Astrodynamics simulation client disconnected.")


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}
_SB3_CACHE: dict[str, "PPO"] = {}

def infer_action_astrodynamics(obs: List[float], model_filename: str | None = None) -> int:
    # If no model filename is provided, try to find the latest one.
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("astrodynamics_policy_") and f.endswith(".zip")]
        if not files:
            print("Warning: No astrodynamics policy found. Returning default action 0 (no thrust).")
            return 0
        files.sort(reverse=True)
        model_filename = files[0]

    model_path = os.path.join(POLICIES_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_filename}' not found. Returning default action 0 (no thrust).")
        return 0

    if model_filename not in _SB3_CACHE:
        try:
            model = PPO.load(model_path)
            _SB3_CACHE[model_filename] = model
        except Exception as e:
            print(f"Error loading SB3 model '{model_filename}': {e}. Returning default action 0.")
            return 0
    
    model = _SB3_CACHE[model_filename]
    action, _ = model.predict(np.array(obs), deterministic=True)
    return int(action)

async def run_astrodynamics(websocket: WebSocket, model_filename: str | None = None):
    env = AstrodynamicsEnv(training_mode=False)
    episode = 0
    obs, _ = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_astrodynamics(list(obs), model_filename)
        nobs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
        state = env.get_state_for_viz()
        await send_json_safely(websocket, {"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.03)
        if done:
            episode += 1
            obs, _ = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0) 