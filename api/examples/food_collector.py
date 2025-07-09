import math
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any, Tuple

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
from fastapi import WebSocket

# -----------------------------------------------------------------------------------
# Food Collector Environment
# -----------------------------------------------------------------------------------

NUM_AGENTS = 5
AREA_SIZE = 40
NUM_GOOD_FOOD = 10
NUM_BAD_FOOD = 3
AGENT_RADIUS = 1.0
FOOD_RADIUS = 0.5
LASER_LENGTH = 25.0
FROZEN_TIME = 4.0

class FoodCollectorEnv:
    def __init__(self, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
        self.width = AREA_SIZE
        self.height = AREA_SIZE
        self.agents_pos = np.zeros((num_agents, 2))
        self.agents_rot = np.zeros(num_agents) # angle in radians
        self.agents_vel = np.zeros((num_agents, 2))
        self.agents_frozen = np.zeros(num_agents, dtype=bool)
        self.agents_frozen_time = np.zeros(num_agents)
        
        self.good_food = np.zeros((NUM_GOOD_FOOD, 2))
        self.bad_food = np.zeros((NUM_BAD_FOOD, 2))

        self.reset()

    def reset(self):
        for i in range(self.num_agents):
            self.agents_pos[i] = np.random.rand(2) * AREA_SIZE
            self.agents_rot[i] = np.random.rand() * 2 * np.pi
        self.agents_vel.fill(0)
        self.agents_frozen.fill(False)
        self.agents_frozen_time.fill(0)
        
        for i in range(NUM_GOOD_FOOD):
            self.good_food[i] = np.random.rand(2) * AREA_SIZE
        for i in range(NUM_BAD_FOOD):
            self.bad_food[i] = np.random.rand(2) * AREA_SIZE
        
        self.steps = 0
        return self._get_all_obs()

    def step(self, actions: List[Tuple[np.ndarray, int]]):
        self.steps += 1
        rewards = np.zeros(self.num_agents)
        
        # --- Update agents based on actions ---
        is_shooting = np.zeros(self.num_agents, dtype=bool)
        for i in range(self.num_agents):
            if self.agents_frozen[i]:
                if self.steps * 0.03 > self.agents_frozen_time[i] + FROZEN_TIME:
                    self.agents_frozen[i] = False
                continue

            cont_actions, disc_action = actions[i]
            
            # Action: 3 continuous, 1 discrete
            # cont[0]: forward, cont[1]: right, cont[2]: rotate
            # disc[0]: shoot laser
            forward_move = cont_actions[0] * 2.0
            side_move = cont_actions[1] * 2.0
            rotation = cont_actions[2] * 3.0

            # Update rotation
            self.agents_rot[i] += rotation * 0.1
            
            # Update velocity
            direction_vec = np.array([np.cos(self.agents_rot[i]), np.sin(self.agents_rot[i])])
            side_vec = np.array([-np.sin(self.agents_rot[i]), np.cos(self.agents_rot[i])])
            
            force = direction_vec * forward_move + side_vec * side_move
            self.agents_vel[i] += force * 0.1
            self.agents_vel[i] *= 0.95 # damping

            # Update position
            self.agents_pos[i] += self.agents_vel[i]

            # Wall collision
            bounce_factor = -0.5
            if self.agents_pos[i, 0] < AGENT_RADIUS:
                self.agents_pos[i, 0] = AGENT_RADIUS
                self.agents_vel[i, 0] *= bounce_factor
            elif self.agents_pos[i, 0] > self.width - AGENT_RADIUS:
                self.agents_pos[i, 0] = self.width - AGENT_RADIUS
                self.agents_vel[i, 0] *= bounce_factor

            if self.agents_pos[i, 1] < AGENT_RADIUS:
                self.agents_pos[i, 1] = AGENT_RADIUS
                self.agents_vel[i, 1] *= bounce_factor
            elif self.agents_pos[i, 1] > self.height - AGENT_RADIUS:
                self.agents_pos[i, 1] = self.height - AGENT_RADIUS
                self.agents_vel[i, 1] *= bounce_factor
            
            if disc_action == 1: # Shoot
                is_shooting[i] = True

        # --- Handle laser shots ---
        for i in range(self.num_agents):
            if is_shooting[i]:
                start_pos = self.agents_pos[i]
                direction = np.array([np.cos(self.agents_rot[i]), np.sin(self.agents_rot[i])])
                end_pos = start_pos + direction * LASER_LENGTH

                for j in range(self.num_agents):
                    if i == j: continue
                    # Simple segment-circle intersection
                    # This is a simplification; true raycast is more complex
                    target_pos = self.agents_pos[j]
                    vec_to_target = target_pos - start_pos
                    proj = vec_to_target.dot(direction)
                    if 0 < proj < LASER_LENGTH:
                        dist_sq = np.sum(vec_to_target**2) - proj**2
                        if dist_sq < AGENT_RADIUS**2:
                            self.agents_frozen[j] = True
                            self.agents_frozen_time[j] = self.steps * 0.03
        
        # --- Food collisions ---
        for i in range(self.num_agents):
            # Good food
            for j in range(NUM_GOOD_FOOD):
                if np.linalg.norm(self.agents_pos[i] - self.good_food[j]) < AGENT_RADIUS + FOOD_RADIUS:
                    rewards[i] += 1.0
                    self.good_food[j] = np.random.rand(2) * AREA_SIZE # respawn
            # Bad food
            for j in range(NUM_BAD_FOOD):
                if np.linalg.norm(self.agents_pos[i] - self.bad_food[j]) < AGENT_RADIUS + FOOD_RADIUS:
                    rewards[i] -= 1.0
                    self.bad_food[j] = np.random.rand(2) * AREA_SIZE # respawn

        done = self.steps > 3000
        dones = [done] * self.num_agents
        
        return self._get_all_obs(is_shooting), rewards, dones

    def _get_obs_for_agent(self, agent_idx, is_shooting):
        agent_pos = self.agents_pos[agent_idx]
        agent_rot = self.agents_rot[agent_idx]
        
        # 1. Local velocity (2)
        # Transform world velocity to agent's local coordinate frame
        world_vel = self.agents_vel[agent_idx]
        cos_rot = np.cos(-agent_rot)
        sin_rot = np.sin(-agent_rot)
        local_vel = np.array([
            world_vel[0] * cos_rot - world_vel[1] * sin_rot,
            world_vel[0] * sin_rot + world_vel[1] * cos_rot
        ])
        
        # 2. Frozen & shoot status (2)
        frozen_status = 1.0 if self.agents_frozen[agent_idx] else 0.0
        shoot_status = 1.0 if is_shooting[agent_idx] else 0.0

        # 3. Grid sensor (7x7=49)
        grid_size = 7
        grid_range = 20.0
        grid = np.zeros((grid_size, grid_size))
        
        # This is a simplified grid sensor. It populates a grid with values
        # indicating objects in range. A real GridSensor is more complex.
        def world_to_grid(pos):
            relative_pos = pos - agent_pos
            # Rotate into agent's reference frame
            x = relative_pos[0] * np.cos(-agent_rot) - relative_pos[1] * np.sin(-agent_rot)
            y = relative_pos[0] * np.sin(-agent_rot) + relative_pos[1] * np.cos(-agent_rot)
            
            if abs(x) > grid_range or abs(y) > grid_range: return None
            
            grid_x = int((x / grid_range * grid_size / 2) + grid_size / 2)
            grid_y = int((y / grid_range * grid_size / 2) + grid_size / 2)
            
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                return grid_x, grid_y
            return None

        # Detect other agents
        for i in range(self.num_agents):
            if i == agent_idx: continue
            coords = world_to_grid(self.agents_pos[i])
            if coords: grid[coords] = 0.25 if self.agents_frozen[i] else 0.5
        
        # Detect food
        for pos in self.good_food:
            coords = world_to_grid(pos)
            if coords: grid[coords] = 1.0
        for pos in self.bad_food:
            coords = world_to_grid(pos)
            if coords: grid[coords] = -1.0
        
        return np.concatenate([
            local_vel,
            [frozen_status, shoot_status],
            grid.flatten()
        ])

    def _get_all_obs(self, is_shooting=None):
        if is_shooting is None:
            is_shooting = np.zeros(self.num_agents, dtype=bool)
        return [self._get_obs_for_agent(i, is_shooting) for i in range(self.num_agents)]

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "agents": [{"pos": pos.tolist(), "rot": float(rot), "frozen": bool(f)} for pos, rot, f in zip(self.agents_pos, self.agents_rot, self.agents_frozen)],
            "good_food": self.good_food.tolist(),
            "bad_food": self.bad_food.tolist(),
            "bounds": [self.width, self.height]
        }

# -----------------------------------------------------------------------------------
# PPO agent (hybrid continuous/discrete actions)
# -----------------------------------------------------------------------------------

OBS_SIZE = 2 + 2 + 49 # vel + status + grid
CONTINUOUS_ACTION_SIZE = 3
DISCRETE_ACTION_SIZE = 2

class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, cont_action_size: int, disc_action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh()
        )
        # Actor heads
        self.cont_mu = nn.Linear(128, cont_action_size)
        self.cont_log_std = nn.Parameter(torch.zeros(1, cont_action_size))
        self.disc_logits = nn.Linear(128, disc_action_size)
        # Critic head
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        # Continuous action distribution
        mu = self.cont_mu(h)
        std = torch.exp(self.cont_log_std.expand_as(mu))
        cont_dist = Normal(mu, std)
        # Discrete action distribution
        logits = self.disc_logits(h)
        disc_dist = Categorical(logits=logits)
        # Critic value
        value = self.critic(h)
        return cont_dist, disc_dist, value

POLICIES_DIR = "policies"
BATCH_SIZE = 4096
MINI_BATCH = 512
EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.005
LR = 3e-4

def _export_model_onnx(model: nn.Module, path: str):
    dummy_input_dim = model.shared[0].in_features
    dummy = torch.zeros((1, dummy_input_dim), dtype=torch.float32)

    class ExportableModel(nn.Module):
        def __init__(self, actor_critic_model):
            super().__init__()
            self.model = actor_critic_model
        def forward(self, obs):
            cont_dist, disc_dist, _ = self.model(obs)
            return cont_dist.mean, disc_dist.logits
    
    exportable_model = ExportableModel(model)
    torch.onnx.export(
        exportable_model, dummy, path,
        input_names=["input"], output_names=["cont_actions", "disc_actions"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "cont_actions": {0: "batch"}, "disc_actions": {0: "batch"}}
    )


# -----------------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------------

async def train_food_collector(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"foodcollector_policy_{ts}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    env = FoodCollectorEnv()
    model = ActorCritic(OBS_SIZE, CONTINUOUS_ACTION_SIZE, DISCRETE_ACTION_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs_list = env.reset()
    obs = torch.tensor(np.array(obs_list), dtype=torch.float32)
    
    ep_counter = 0
    step_buffer: list[dict] = []
    num_envs = env.num_agents # Treat each agent as an "env"

    total_episodes = 25000

    while ep_counter < total_episodes:
        with torch.no_grad():
            cont_dist, disc_dist, value = model(obs)
            cont_actions = cont_dist.sample()
            disc_actions = disc_dist.sample()
            
            cont_logp = cont_dist.log_prob(cont_actions).sum(dim=1)
            disc_logp = disc_dist.log_prob(disc_actions)
            logp = (cont_logp + disc_logp).unsqueeze(1)

        cont_actions_np = cont_actions.cpu().numpy()
        disc_actions_np = disc_actions.cpu().numpy()
        
        actions_list = list(zip(cont_actions_np, disc_actions_np))
        next_obs_list, rewards, dones = env.step(actions_list)
        
        next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        step_buffer.append({
            "obs": obs, "cont_actions": cont_actions, "disc_actions": disc_actions,
            "logp": logp, "reward": rewards_t, "done": dones_t, "value": value
        })
        obs = next_obs

        if len(step_buffer) % 4 == 0:
            state = env.get_state_for_viz()
            await websocket.send_json({"type": "train_step", "state": state, "episode": ep_counter})
            await asyncio.sleep(0.001)

        if any(dones):
            ep_counter += 1
            # In this setup, all agents "finish" at the same time
            # No individual resets needed, just wait for batch to process
        
        if len(step_buffer) * num_envs >= BATCH_SIZE:
            # --- LR Annealing ---
            frac = 1.0 - (ep_counter / total_episodes)
            new_lr = LR * frac
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

            with torch.no_grad():
                _, _, next_value = model(obs)

            # --- PPO Update ---
            num_steps = len(step_buffer)
            values = torch.cat([b["value"] for b in step_buffer]).view(num_steps, num_envs)
            rewards = torch.cat([b["reward"] for b in step_buffer]).view(num_steps, num_envs)
            dones = torch.cat([b["done"] for b in step_buffer]).view(num_steps, num_envs)
            all_values = torch.cat([values, next_value.view(1, num_envs)], dim=0)

            advantages = torch.zeros(num_steps, num_envs)
            gae = 0.0
            for t in reversed(range(num_steps)):
                delta = rewards[t] + GAMMA * (1.0 - dones[t]) * all_values[t + 1] - all_values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - dones[t]) * gae
                advantages[t] = gae
            
            adv = advantages.flatten().unsqueeze(1)
            returns = adv + values.flatten().unsqueeze(1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            obs_cat = torch.cat([b["obs"] for b in step_buffer])
            cont_act_cat = torch.cat([b["cont_actions"] for b in step_buffer])
            disc_act_cat = torch.cat([b["disc_actions"] for b in step_buffer]).unsqueeze(1)
            logp_cat = torch.cat([b["logp"] for b in step_buffer])

            for _ in range(EPOCHS):
                idx = torch.randperm(obs_cat.shape[0])
                for start in range(0, obs_cat.shape[0], MINI_BATCH):
                    mb_idx = idx[start:start + MINI_BATCH]
                    mb_obs = obs_cat[mb_idx]
                    
                    cont_dist, disc_dist, value = model(mb_obs)
                    cont_logp = cont_dist.log_prob(cont_act_cat[mb_idx]).sum(dim=1)
                    disc_logp = disc_dist.log_prob(disc_act_cat[mb_idx].squeeze())
                    logp_new = (cont_logp + disc_logp).unsqueeze(1)
                    
                    entropy_bonus = (cont_dist.entropy().sum(dim=1) + disc_dist.entropy()).mean()
                    
                    ratio = (logp_new - logp_cat[mb_idx]).exp()
                    
                    mb_adv = adv[mb_idx]
                    policy_loss1 = ratio * mb_adv
                    policy_loss2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
                    
                    value_loss = (value - returns[mb_idx]).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - ENT_COEF * entropy_bonus

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

            avg_reward = float(torch.cat([b["reward"] for b in step_buffer]).mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            await websocket.send_json({"type": "progress", "episode": ep_counter, "reward": avg_reward, "loss": avg_loss})
    
    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": ts, "session_uuid": session_uuid})


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}

def infer_action_food_collector(obs: List[float], model_filename: str | None = None) -> Tuple[np.ndarray, int]:
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("foodcollector_policy_") and f.endswith(".onnx")]
        if not files: raise FileNotFoundError("No foodcollector policy found.")
        files.sort(reverse=True)
        model_filename = files[0]
    
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess

    inp = np.array([obs], dtype=np.float32)
    cont_out, disc_out = _ORT_CACHE[model_filename].run(None, {"input": inp})
    
    disc_action = np.argmax(disc_out, axis=1)[0]
    return cont_out[0], int(disc_action)


async def run_food_collector(websocket: WebSocket, model_filename: str | None = None):
    env = FoodCollectorEnv()
    from starlette.websockets import WebSocketState
    
    obs_list = env.reset()
    while websocket.application_state == WebSocketState.CONNECTED:
        actions_list = [infer_action_food_collector(obs, model_filename) for obs in obs_list]
        
        obs_list, _, dones = env.step(actions_list)
        
        state = env.get_state_for_viz()
        await websocket.send_json({"type": "run_step", "state": state})
        await asyncio.sleep(0.03)

        if any(dones):
            obs_list = env.reset()
        
        await asyncio.sleep(0.01) 