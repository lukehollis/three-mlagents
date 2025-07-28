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
from fastapi import WebSocket

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
TOTAL_TIMESTEPS = 12000 * EPISODES # Corresponds to old timeout logic

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
        self.max_thrust = 500000.0  # N (powerful, but allows for sustained burn)
        self.specific_impulse = 300000.0  # seconds
        self.initial_fuel = 500000.0  # kg (enough for a long burn)
        self.dt = 50.0  # seconds (more stable timestep)
        
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
        
        self.action_space = spaces.Discrete(ACTION_SIZE)
        obs_shape = self._get_obs_dummy().shape
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

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
        
        # Calculate initial relative state
        self.relative_pos = self.spacecraft_pos_abs - self.target_pos_abs
        self.relative_vel = self.spacecraft_vel_abs - self.target_vel_abs
        
        self.fuel = self.initial_fuel
        self.spacecraft_trail.clear()
        self.target_trail.clear()
        self.steps = 0
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

        # Calculate gravitational acceleration for both spacecraft and target
        spacecraft_grav_accel = self._orbital_dynamics(self.spacecraft_pos_abs)
        target_grav_accel = self._orbital_dynamics(self.target_pos_abs)

        # Total acceleration for spacecraft
        thrust_accel = np.zeros(3)
        current_mass = self.mass + self.fuel
        if self.fuel > 0 and current_mass > 0:
            thrust_accel = thrust_vector / current_mass
        
        spacecraft_total_accel = spacecraft_grav_accel + thrust_accel

        # Integrate motion for spacecraft
        self.spacecraft_vel_abs += spacecraft_total_accel * self.dt
        self.spacecraft_pos_abs += self.spacecraft_vel_abs * self.dt
        
        # Integrate motion for target
        self.target_vel_abs += target_grav_accel * self.dt
        self.target_pos_abs += self.target_vel_abs * self.dt
        
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

        if not self.training_mode:
            if spacecraft_radius < self.earth_radius:
                terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        # Universal Failure & Success Conditions
        if spacecraft_radius < self.earth_radius:
            print(f"Episode End: Crashed into Earth at step {self.steps}.")
            reward = -200.0; terminated = True
        elif distance > self.max_distance:
            print(f"Episode End: Exceeded max distance at step {self.steps}.")
            reward = -10.0; terminated = True
        elif self.fuel <= 0 and distance > self.docking_threshold:
            print(f"Episode End: Out of fuel at step {self.steps}.")
            reward = -50.0; terminated = True
        elif distance < self.docking_threshold and velocity_mag > self.velocity_threshold:
            print(f"Episode End: Crashed into target at step {self.steps}.")
            reward = -50.0; terminated = True
        elif self.steps > 12000:
            print(f"Episode End: Timeout at step {self.steps}.")
            reward = -5.0; truncated = True
        elif distance < self.docking_threshold and velocity_mag < self.velocity_threshold:
            print(f"Episode End: Successfully docked at step {self.steps}.")
            reward = 1000.0; terminated = True
        
        if terminated or truncated:
            return self._get_obs(), reward, terminated, truncated, {}

        # --- State-Dependent Reward Logic ---
        # The agent MUST be forced to leave LEO immediately.
        LEO_EXIT_THRESHOLD_RADIUS = self.leo_radius * 1.1

        if spacecraft_radius < LEO_EXIT_THRESHOLD_RADIUS:
            # PHASE 1: LEAVING LEO.
            # The only goal is to thrust. Any thrust.
            if action == 0:
                # Unignorable penalty for inaction.
                reward = -100.0
            else:
                # Positive reward for ANY thrust action.
                reward = 5.0
        
        elif distance > self.docking_approach_threshold:
            # PHASE 2: ORBITAL RENDEZVOUS.
            # Now that we've left LEO, we can use more nuanced rewards.
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
            # PHASE 3: DOCKING (Terminal Guidance)
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
        
        return self._get_obs(), reward, terminated, truncated, {}

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
            [self.steps / 6000.0]              # Time progress (1)
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
    def __init__(self, websocket: WebSocket, verbose=0):
        super(WebSocketCallback, self).__init__(verbose)
        self.websocket = websocket
        self.loop = asyncio.get_event_loop()

    def _on_step(self) -> bool:
        # Send visualization state periodically for a smooth animation.
        if self.num_timesteps % 128 == 0:
            states = self.training_env.env_method("get_state_for_viz", indices=[0])
            if states:
                state = states[0]
                payload = {"type": "state", "state": state}
                asyncio.run_coroutine_threadsafe(self.websocket.send_json(payload), self.loop)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered after collecting a rollout and before updating the policy.
        The logger has access to the latest rollout metrics here.
        """
        reward = self.logger.name_to_value.get("rollout/ep_rew_mean")
        loss = self.logger.name_to_value.get("train/loss")

        # Only send progress if a reward value is present.
        if reward is not None:
            payload = {
                "type": "progress",
                "episode": self.num_timesteps,  # Use timesteps for the x-axis
                "reward": float(reward),
                "loss": float(loss) if loss is not None else 0.0,
            }
            asyncio.run_coroutine_threadsafe(self.websocket.send_json(payload), self.loop)


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

async def train_astrodynamics(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename_base = f"astrodynamics_policy_{ts}_{session_uuid}"
    model_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.zip")
    onnx_path = os.path.join(POLICIES_DIR, f"{model_filename_base}.onnx")
    
    # Use a vectorized environment for parallel training
    # Recommended to use 'fork' start method on Linux for performance
    # 'spawn' is default on macOS and Windows, which is fine.
    vec_env = make_vec_env(AstrodynamicsEnv, n_envs=32, env_kwargs=dict(training_mode=True))

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
    )

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

    callback = WebSocketCallback(websocket)
    
    # model.learn is a blocking call, so run it in a thread
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=True
        )
    )

    model.save(model_path)
    _export_model_onnx(model, onnx_path)
    
    await websocket.send_json({
        "type": "trained",
        "file_url": f"/policies/{model_filename_base}.zip",
        "model_filename": f"{model_filename_base}.zip",
        "onnx_filename": f"{model_filename_base}.onnx",
        "timestamp": ts,
        "session_uuid": session_uuid
    })


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
            await websocket.send_json({"type": "state", "state": state})
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
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.05)
        if done:
            episode += 1
            obs, _ = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0) 