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
import torch.optim as optim
from torch.distributions.categorical import Categorical
from fastapi import WebSocket


ACTION_SIZE = 7  # 0=no thrust, 1-6=thrust directions
POLICIES_DIR = "policies"
BATCH_SIZE = 4096
MINI_BATCH = 512
EPOCHS = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
ENT_COEF = 0.1
LR = 5e-4
EPISODES = 3000

# -----------------------------------------------------------------------------------
# Astrodynamics Environment
# -----------------------------------------------------------------------------------

class AstrodynamicsEnv:
    """An orbital rendezvous environment for spacecraft navigation."""

    def __init__(self, training_mode: bool = True):
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
        self.max_distance = 25000e5  # maximum distance from target (m)
        
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
        self.reset()

    def _orbital_dynamics(self, pos_global: np.ndarray) -> np.ndarray:
        """Calculate gravitational acceleration at given global position."""
        r = np.linalg.norm(pos_global)
        if r < self.earth_radius:
            r = self.earth_radius  # Prevent division by zero
        
        # Gravitational acceleration vector pointing toward Earth center
        accel = -self.mu * pos_global / (r**3)
        return accel

    def reset(self):
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
        return self._get_obs()

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
        done = False
        reward = 0.0

        if not self.training_mode:
            if spacecraft_radius < self.earth_radius: done = True
            return self._get_obs(), reward, done

        # Universal Failure & Success Conditions
        if spacecraft_radius < self.earth_radius:
            print(f"Episode End: Crashed into Earth at step {self.steps}.")
            reward = -200.0; done = True
        elif distance > self.max_distance:
            print(f"Episode End: Exceeded max distance at step {self.steps}.")
            reward = -10.0; done = True
        elif self.fuel <= 0 and distance > self.docking_threshold:
            print(f"Episode End: Out of fuel at step {self.steps}.")
            reward = -50.0; done = True
        elif distance < self.docking_threshold and velocity_mag > self.velocity_threshold:
            print(f"Episode End: Crashed into target at step {self.steps}.")
            reward = -50.0; done = True
        elif self.steps > 12000:
            print(f"Episode End: Timeout at step {self.steps}.")
            reward = -5.0; done = True
        elif distance < self.docking_threshold and velocity_mag < self.velocity_threshold:
            print(f"Episode End: Successfully docked at step {self.steps}.")
            reward = 1000.0; done = True
        
        if done:
            return self._get_obs(), reward, done

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
        else:
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

            # 3. Approach Reward
            distance_penalty = (distance / self.max_distance) * 5.0
            reward -= altitude_proximity * distance_penalty

        # Small universal time penalty
        reward -= 0.1
        
        return self._get_obs(), reward, done

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
# PPO agent (adapted for discrete actions)
# -----------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_logits = nn.Linear(128, action_size)
        self.actor_logits.bias.data.fill_(0.0)
        self.actor_logits.bias.data[0] = -1.0  # Slight bias against action 0 (no thrust)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value


def _export_model_onnx(model: nn.Module, path: str):
    dummy_input_dim = model.shared[0].in_features
    dummy = torch.zeros((1, dummy_input_dim), dtype=torch.float32)
    class ExportableModel(nn.Module):
        def __init__(self, actor_critic_model):
            super().__init__()
            self.model = actor_critic_model
        def forward(self, obs):
            dist, _ = self.model(obs)
            return dist.logits
    
    exportable_model = ExportableModel(model)
    torch.onnx.export(
        exportable_model, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )

# -----------------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------------

async def train_astrodynamics(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"astrodynamics_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[AstrodynamicsEnv] = [AstrodynamicsEnv() for _ in range(32)]
    OBS_SIZE = envs[0]._get_obs().shape[0]

    model = ActorCritic(OBS_SIZE, ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor(np.array([e.reset() for e in envs]), dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    
    total_steps = 0

    while ep_counter < EPISODES: # Increased training episodes
        with torch.no_grad():
            dist, value = model(obs)
            actions = dist.sample()
            logp = dist.log_prob(actions).unsqueeze(1)
        
        actions_np = actions.cpu().numpy()

        step_obs, rewards, dones = [], [], []
        for idx, env in enumerate(envs):
            nobs, rew, dn = env.step(actions_np[idx])
            step_obs.append(nobs)
            rewards.append(rew)
            dones.append(dn)
        
        total_steps += len(envs)
        next_obs = torch.tensor(np.array(step_obs), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        step_buffer.append({"obs": obs, "actions": actions.unsqueeze(1), "logp": logp, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        if len(step_buffer) % 16 == 0:
            state = envs[0].get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter + 1})
            await asyncio.sleep(0.01)

        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                obs[i] = torch.tensor(envs[i].reset(), dtype=torch.float32)

        if total_steps >= BATCH_SIZE:
            with torch.no_grad():
                _, next_value = model(obs)
                next_value = next_value.squeeze()

            num_steps = len(step_buffer)
            num_envs = len(envs)
            
            values = torch.cat([b["value"] for b in step_buffer]).view(num_steps, num_envs)
            rewards = torch.cat([b["reward"] for b in step_buffer]).view(num_steps, num_envs)
            dones = torch.cat([b["done"] for b in step_buffer]).view(num_steps, num_envs)
            all_values = torch.cat([values, next_value.unsqueeze(0)], dim=0)

            advantages = torch.zeros(num_steps, num_envs)
            gae = 0.0
            for t in reversed(range(num_steps)):
                delta = rewards[t] + GAMMA * (1.0 - dones[t]) * all_values[t + 1] - all_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
                advantages[t] = gae
            
            adv = advantages.flatten().unsqueeze(1)
            val_cat = values.flatten().unsqueeze(1)
            returns = adv + val_cat
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            obs_cat = torch.cat([b["obs"] for b in step_buffer])
            act_cat = torch.cat([b["actions"] for b in step_buffer])
            logp_cat = torch.cat([b["logp"] for b in step_buffer])

            for _ in range(EPOCHS):
                idx = torch.randperm(obs_cat.shape[0])
                for start in range(0, obs_cat.shape[0], MINI_BATCH):
                    mb_idx = idx[start:start + MINI_BATCH]
                    mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret = obs_cat[mb_idx], act_cat[mb_idx], logp_cat[mb_idx], adv[mb_idx], returns[mb_idx]

                    dist, value = model(mb_obs)
                    logp_new = dist.log_prob(mb_act.squeeze(1)).unsqueeze(1)
                    entropy_bonus = dist.entropy().mean()
                    ratio = (logp_new - mb_logp_old).exp()
                    
                    policy_loss1 = ratio * mb_adv
                    policy_loss2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                    
                    value_loss = (value - mb_ret).pow(2).mean()
                    approx_kl = (mb_logp_old - logp_new).mean()
                    if approx_kl > 0.01:
                        continue
                    value_clipped = val_cat[mb_idx] + torch.clamp(value - val_cat[mb_idx], -CLIP_EPS, CLIP_EPS)
                    value_loss_clipped = (value_clipped - mb_ret).pow(2).mean()
                    value_loss = 0.5 * torch.max(value_loss, value_loss_clipped)

                    loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            avg_reward = float(torch.cat([b["reward"] for b in step_buffer]).mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            total_steps = 0
            if ep_counter > 10: # Avoid noisy first updates
                await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": ts, "session_uuid": session_uuid})

async def run_simulation(websocket: WebSocket):
    """Runs a physics-only simulation and streams state."""
    env = AstrodynamicsEnv(training_mode=False)
    env.reset()
    from starlette.websockets import WebSocketState, WebSocketDisconnect

    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            # Action 0 is no thrust, simulating orbital mechanics
            _, _, done = env.step(0)
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

def infer_action_astrodynamics(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    
    # If no model filename is provided, try to find the latest one.
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("astrodynamics_policy_") and f.endswith(".onnx")]
        if not files:
            # If no policy is found, return a default action (e.g., no thrust)
            print("Warning: No astrodynamics policy found. Returning default action 0 (no thrust).")
            return 0
        files.sort(reverse=True)
        model_filename = files[0]

    model_path = os.path.join(POLICIES_DIR, model_filename)

    # Check if the specific model file exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_filename}' not found. Returning default action 0 (no thrust).")
        return 0
    
    if model_filename not in _ORT_CACHE:
        try:
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            _ORT_CACHE[model_filename] = sess
        except Exception as e:
            print(f"Error loading ONNX model '{model_filename}': {e}. Returning default action 0.")
            return 0

    try:
        inp = np.array([obs], dtype=np.float32)
        out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
        action = np.argmax(out, axis=1)[0]
        return int(action)
    except Exception as e:
        print(f"Error during inference with model '{model_filename}': {e}. Returning default action 0.")
        return 0

async def run_astrodynamics(websocket: WebSocket, model_filename: str | None = None):
    env = AstrodynamicsEnv(training_mode=False)
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_astrodynamics(list(obs), model_filename)
        nobs, _, done = env.step(act)
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.05)
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0) 