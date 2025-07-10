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
        self.CL_alpha = 2 * np.pi # Lift coefficient slope
        self.CD0 = 0.02 # Parasitic drag
        self.CD_k = 0.05 # Induced drag factor
        self.dt = 0.02

        # Waypoints for navigation task
        self.waypoints = [np.array([-80.0, 0.0, 70.0]), np.array([80.0, 0.0, 70.0])]
        self.current_waypoint_index = 0
        self.waypoint_threshold = 15.0 # How close to get to a waypoint

        # Wind model (sigmoid profile from the paper)
        self.wind_C1 = 15.0  # max wind speed
        self.wind_C2 = 0.1   # gradient thickness
        self.wind_C3 = 50.0  # height of max gradient
        self.wind_wave_freq = 1.0 / 70.0  # spatial frequency of y-variation
        self.wind_wave_mag = np.pi / 9    # magnitude of y-variation in radians (+/- 20 deg)
        self.wind_wave_freq2 = 1.0 / 250.0 # spatial frequency of a second wave
        self.wind_wave_mag2 = np.pi / 6   # magnitude of second wave (+/- 30 deg)

        # State variables
        self.pos = np.zeros(3)  # x, y, z
        self.vel = np.zeros(3)  # vx, vy, vz
        self.rot = np.zeros(3)  # roll, pitch, yaw (phi, theta, psi)
        self.ang_vel = np.zeros(3) # p, q, r

        self.max_roll = np.pi / 2
        self.max_pitch = np.pi / 4
        self.max_aoa = np.deg2rad(15)

        self.steps = 0
        self.reset()

    def _wind_model(self, pos: np.ndarray) -> np.ndarray:
        # Wind blows along positive x direction, with sinusoidal variance in y
        z, y = pos[2], pos[1]
        
        # Base sigmoid wind profile based on height
        base_wind_speed = self.wind_C1 / (1 + np.exp(-self.wind_C2 * (z - self.wind_C3)))
        
        # Superimpose two sine waves for a more complex, non-uniform pattern
        angle_variation1 = np.sin(y * self.wind_wave_freq * 2 * np.pi) * self.wind_wave_mag
        angle_variation2 = np.sin(y * self.wind_wave_freq2 * 2 * np.pi) * self.wind_wave_mag2
        total_angle_variation = angle_variation1 + angle_variation2
        
        wind_dir_x = np.cos(total_angle_variation)
        wind_dir_y = np.sin(total_angle_variation)
        
        return np.array([base_wind_speed * wind_dir_x, base_wind_speed * wind_dir_y, 0.0])

    def reset(self):
        self.pos = np.array([0.0, 0.0, 60.0])
        self.vel = np.array([20.0, 0.0, -1.0])
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
            yaw_torque = 4.0 # Coordinated turn
        elif action == 2: 
            roll_torque = 15.0
            yaw_torque = -4.0 # Coordinated turn
        elif action == 3: pitch_torque = 10.0
        elif action == 4: pitch_torque = -10.0
        
        # --- Physics update ---
        # Simplified rotational dynamics
        self.ang_vel[0] += roll_torque * self.dt
        self.ang_vel[1] += pitch_torque * self.dt
        self.ang_vel[2] += yaw_torque * self.dt
        self.ang_vel *= 0.95 # damping
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
            
            # Rotate forces to world frame (crude approximation)
            R_roll = np.array([[1, 0, 0], [0, np.cos(self.rot[0]), -np.sin(self.rot[0])], [0, np.sin(self.rot[0]), np.cos(self.rot[0])]])
            R_pitch = np.array([[np.cos(self.rot[1]), 0, np.sin(self.rot[1])], [0, 1, 0], [-np.sin(self.rot[1]), 0, np.cos(self.rot[1])]])
            R = R_pitch @ R_roll # Yaw not affecting lift/drag direction much
            
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
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)

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

        # Penalty for crashing
        if self.pos[2] < 5.0:
            reward = -50.0
            done = True
        
        # Penalty for stalling
        if abs(aoa) > self.max_aoa:
            reward = -50.0
            done = True

        # Penalty for flying too far away from the action
        if dist_to_target > 300: # If it's more than 2x the waypoint separation
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

        return np.concatenate([
            np.array([
                self.vel[2] / 10.0,            # vertical speed (normalized)
                (self.pos[2] - self.wind_C3) / 50.0, # height relative to wind layer (normalized)
                self.rot[0],                   # roll
                self.rot[1],                   # pitch
                self.ang_vel[0],               # roll rate
                self.ang_vel[1],               # pitch rate
            ]),
            self.vel / 20.0, # velocity (normalized)
            dir_to_target, # direction to target
            [dist_to_target / 100.0] # distance to target (normalized)
        ])

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "pos": self.pos.tolist(),
            "rot": self.rot.tolist(), # roll, pitch, yaw
            "wind_params": [self.wind_C1, self.wind_C2, self.wind_C3, self.wind_wave_freq, self.wind_wave_mag, self.wind_wave_freq2, self.wind_wave_mag2],
            "bounds": [200, 200],
            "waypoints": [w.tolist() for w in self.waypoints],
            "current_waypoint_index": self.current_waypoint_index
        }

# -----------------------------------------------------------------------------------
# PPO agent (adapted for discrete actions)
# -----------------------------------------------------------------------------------

ACTION_SIZE = 5

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

async def train_glider(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"glider_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[GliderEnv] = [GliderEnv() for _ in range(16)]
    OBS_SIZE = envs[0]._get_obs().shape[0]

    model = ActorCritic(OBS_SIZE, ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor(np.array([e.reset() for e in envs]), dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    
    total_steps = 0

    while ep_counter < 3000:
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
            await asyncio.sleep(0.00)

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

def infer_action_glider(obs: List[float], model_filename: str | None = None) -> int:
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("glider_policy_") and f.endswith(".onnx")]
        if not files:
            raise FileNotFoundError("No glider policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    action = np.argmax(out, axis=1)[0]
    return int(action)


async def run_glider(websocket: WebSocket, model_filename: str | None = None):
    env = GliderEnv()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act = infer_action_glider(list(obs), model_filename)
        nobs, _, done = env.step(act)
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state, "episode": episode + 1})
        await asyncio.sleep(0.03)
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0) 