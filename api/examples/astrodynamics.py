import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# Astrodynamics Environment
# -----------------------------------------------------------------------------------

class AstrodynamicsEnv:
    """An orbital rendezvous environment for spacecraft navigation."""

    def __init__(self):
        # Physical constants
        self.mu = 3.986e14  # Earth's gravitational parameter (m³/s²)
        self.earth_radius = 6.371e6  # Earth radius (m)
        self.orbit_altitude = 400e3  # Orbital altitude (m)
        self.orbit_radius = self.earth_radius + self.orbit_altitude
        
        # Orbital velocity at this altitude
        self.orbital_velocity = np.sqrt(self.mu / self.orbit_radius)
        
        # Spacecraft parameters
        self.mass = 1000.0  # kg
        self.max_thrust = 500.0  # N
        self.specific_impulse = 300.0  # seconds
        self.initial_fuel = 500.0  # kg
        self.dt = 1.0  # seconds
        
        # Mission parameters
        self.docking_threshold = 10.0  # meters
        self.velocity_threshold = 0.5  # m/s for successful docking
        self.max_distance = 5000.0  # maximum distance from target (m)
        
        # State variables (in orbital reference frame)
        self.pos = np.zeros(3)  # relative position to target (m)
        self.vel = np.zeros(3)  # relative velocity to target (m/s)
        self.fuel = self.initial_fuel  # remaining fuel (kg)
        self.target_pos = np.zeros(3)  # target space station position
        self.target_vel = np.zeros(3)  # target space station velocity
        
        # Visualization trail
        self.spacecraft_trail = []
        self.max_trail_length = 100
        
        self.steps = 0
        self.reset()

    def _orbital_dynamics(self, pos_global: np.ndarray) -> np.ndarray:
        """Calculate gravitational acceleration at given global position."""
        r = np.linalg.norm(pos_global)
        if r < self.earth_radius:
            r = self.earth_radius  # Prevent division by zero
        
        # Gravitational acceleration vector pointing toward Earth center
        accel = -self.mu * pos_global / (r**3)
        return accel

    def _relative_motion_dynamics(self, rel_pos: np.ndarray, rel_vel: np.ndarray) -> tuple:
        """Hill's equations for relative orbital motion."""
        x, y, z = rel_pos
        vx, vy, vz = rel_vel
        
        # Mean motion (angular velocity of reference orbit)
        n = np.sqrt(self.mu / (self.orbit_radius**3))
        
        # Hill's equations (Clohessy-Wiltshire)
        ax = 3 * n**2 * x + 2 * n * vy
        ay = -2 * n * vx
        az = -n**2 * z
        
        return np.array([ax, ay, az])

    def reset(self):
        # Start spacecraft at a random position around the target
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(1000, 3000)  # meters
        
        self.pos = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            np.random.uniform(-500, 500)
        ])
        
        # Small initial relative velocity
        self.vel = np.random.uniform(-2, 2, 3)
        
        # Target is at origin of relative coordinate system
        self.target_pos = np.array([0.0, 0.0, 0.0])
        self.target_vel = np.array([0.0, 0.0, 0.0])
        
        self.fuel = self.initial_fuel
        self.spacecraft_trail = []
        self.steps = 0
        return self._get_obs()

    def step(self, action: int):
        self.steps += 1

        # Action mapping: 0=no thrust, 1-6=thrust in +/-x,y,z directions
        thrust_vector = np.zeros(3)
        thrust_magnitude = self.max_thrust
        
        if action == 1:   # +X thrust
            thrust_vector = np.array([thrust_magnitude, 0, 0])
        elif action == 2: # -X thrust
            thrust_vector = np.array([-thrust_magnitude, 0, 0])
        elif action == 3: # +Y thrust
            thrust_vector = np.array([0, thrust_magnitude, 0])
        elif action == 4: # -Y thrust
            thrust_vector = np.array([0, -thrust_magnitude, 0])
        elif action == 5: # +Z thrust
            thrust_vector = np.array([0, 0, thrust_magnitude])
        elif action == 6: # -Z thrust
            thrust_vector = np.array([0, 0, -thrust_magnitude])
        
        # Calculate fuel consumption
        thrust_mag = np.linalg.norm(thrust_vector)
        if thrust_mag > 0:
            fuel_consumed = thrust_mag * self.dt / (self.specific_impulse * 9.81)
            self.fuel = max(0, self.fuel - fuel_consumed)
            
            # Reduce thrust if out of fuel
            if self.fuel <= 0:
                thrust_vector = np.zeros(3)

        # Apply orbital dynamics
        rel_accel = self._relative_motion_dynamics(self.pos, self.vel)
        
        # Add thrust acceleration
        if self.fuel > 0:
            thrust_accel = thrust_vector / self.mass
            rel_accel += thrust_accel
        
        # Integrate motion
        self.vel += rel_accel * self.dt
        self.pos += self.vel * self.dt
        
        # Update trail for visualization
        self.spacecraft_trail.append(self.pos.copy())
        if len(self.spacecraft_trail) > self.max_trail_length:
            self.spacecraft_trail.pop(0)

        # Calculate reward and termination
        distance = np.linalg.norm(self.pos)
        velocity_mag = np.linalg.norm(self.vel)
        
        done = False
        reward = 0.0
        
        # Primary reward: negative distance (encourages approach)
        reward -= distance / 1000.0  # Scale to reasonable range
        
        # Velocity penalty (encourages slow, controlled approach)
        reward -= velocity_mag * 0.1
        
        # Fuel efficiency bonus
        fuel_ratio = self.fuel / self.initial_fuel
        reward += fuel_ratio * 0.01
        
        # Success reward
        if distance < self.docking_threshold and velocity_mag < self.velocity_threshold:
            reward += 100.0  # Large success bonus
            done = True
        
        # Failure conditions
        if distance > self.max_distance:
            reward = -50.0  # Penalty for drifting too far
            done = True
        
        if self.fuel <= 0 and distance > self.docking_threshold:
            reward = -30.0  # Penalty for running out of fuel
            done = True
        
        # Collision penalty (too fast approach)
        if distance < self.docking_threshold and velocity_mag > self.velocity_threshold:
            reward = -25.0
            done = True
        
        # Timeout
        if self.steps > 1000:
            done = True

        return self._get_obs(), reward, done

    def _get_obs(self):
        """Get observation vector for the agent."""
        distance = np.linalg.norm(self.pos)
        velocity_mag = np.linalg.norm(self.vel)
        
        # Normalize observations
        pos_norm = self.pos / 5000.0  # Normalize by max distance
        vel_norm = self.vel / 10.0    # Normalize by reasonable velocity range
        
        # Unit vector pointing to target
        target_dir = -self.pos / (distance + 1e-8)
        
        return np.concatenate([
            pos_norm,                          # Relative position (3)
            vel_norm,                          # Relative velocity (3)
            target_dir,                        # Direction to target (3)
            [distance / 5000.0],               # Distance to target (1)
            [velocity_mag / 10.0],             # Speed magnitude (1)
            [self.fuel / self.initial_fuel],   # Fuel ratio (1)
            [self.steps / 1000.0]              # Time progress (1)
        ])

    def get_state_for_viz(self) -> Dict[str, Any]:
        """Get state information for visualization."""
        return {
            "spacecraft_pos": self.pos.tolist(),
            "spacecraft_vel": self.vel.tolist(),
            "target_pos": self.target_pos.tolist(),
            "fuel_ratio": self.fuel / self.initial_fuel,
            "distance_to_target": float(np.linalg.norm(self.pos)),
            "velocity_magnitude": float(np.linalg.norm(self.vel)),
            "trail": [pos.tolist() for pos in self.spacecraft_trail],
            "orbit_params": {
                "radius": self.orbit_radius,
                "velocity": self.orbital_velocity
            }
        }

# -----------------------------------------------------------------------------------
# PPO agent (adapted for discrete actions)
# -----------------------------------------------------------------------------------

ACTION_SIZE = 7  # 0=no thrust, 1-6=thrust directions

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        self.actor_logits = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits = self.actor_logits(h)
        dist = Categorical(logits=logits)
        value = self.critic(h)
        return dist, value

POLICIES_DIR = "policies"
BATCH_SIZE = 4096
MINI_BATCH = 512
EPOCHS = 8
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-4

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

    envs: List[AstrodynamicsEnv] = [AstrodynamicsEnv() for _ in range(16)]
    OBS_SIZE = envs[0]._get_obs().shape[0]

    model = ActorCritic(OBS_SIZE, ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor(np.array([e.reset() for e in envs]), dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    
    total_steps = 0

    while ep_counter < 2000:
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
                    loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            avg_reward = float(torch.cat([b["reward"] for b in step_buffer]).mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            total_steps = 0
            await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": ts, "session_uuid": session_uuid})

# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}

def infer_action_astrodynamics(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("astrodynamics_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No astrodynamics policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    action = np.argmax(out, axis=1)[0]
    return int(action)

async def run_astrodynamics(websocket: WebSocket, model_filename: str | None = None):
    env = AstrodynamicsEnv()
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