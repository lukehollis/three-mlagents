# --------------------------------------------------------------
# Crawler Example – MuJoCo implementation with proper 2 DOF joints
# --------------------------------------------------------------
# This implementation uses MuJoCo for accurate physics simulation,
# matching Unity ML-Agents Crawler behavior with true 2 DOF upper leg joints.

import os
import math
import asyncio
from datetime import datetime
import uuid
from typing import List
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket
import mujoco
import mujoco.viewer

# -----------------------------------------------------------------------------------
# Timing Configuration for Observation
# -----------------------------------------------------------------------------------
# For FASTEST training: Keep ENABLE_TRAINING_DELAYS = False
# For OBSERVABLE training: Set ENABLE_TRAINING_DELAYS = True and adjust delays below
ENABLE_TRAINING_DELAYS = False  # Set to True for slower, observable training
TRAINING_DELAY = 0.02    # Reduced from 0.1 for faster training
INFERENCE_DELAY = 0.03   # Reduced from 0.05 for slightly faster inference
TRAINING_STREAM_FREQ = 8  # Increased from 4 to reduce update frequency

# -----------------------------------------------------------------------------------
# MJCF Model Definition
# -----------------------------------------------------------------------------------

CRAWLER_XML = """
<mujoco model="crawler">
  <compiler angle="degree" inertiafromgeom="true"/>
  
  <option gravity="0 0 -9.81" timestep="0.02"/>
  
  <default>
    <joint damping="0.1" armature="0.1"/>
    <geom contype="1" conaffinity="1" friction="1 0.1 0.1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  
  <asset>
    <material name="blue" rgba="0 0.5 1 1"/>
    <material name="orange" rgba="1 0.5 0 1"/>
    <material name="yellow" rgba="1 0.8 0.2 1"/>
    <material name="red" rgba="1 0.2 0.2 1"/>
    <material name="ground" rgba="0.8 0.8 0.8 1"/>
  </asset>
  
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="50 50 0.1" material="ground"/>
    
    <!-- Torso (free floating) -->
    <body name="torso" pos="0 0 0.25">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="0.3 0.2 0.08" material="blue"/>
      <site name="torso_site" pos="0 0 0" size="0.01"/>
      
      <!-- Front Left Leg -->
      <body name="fl_upper" pos="0.25 0.15 -0.05">
        <joint name="fl_hip_x" type="hinge" axis="1 0 0" range="-60 0" damping="3"/>
        <joint name="fl_hip_y" type="hinge" axis="0 1 0" range="-20 20" damping="3"/>
        <geom name="fl_upper" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="orange"/>
        
        <body name="fl_lower" pos="0 0 -0.12">
          <joint name="fl_knee" type="hinge" axis="1 0 0" range="0 150" damping="2"/>
          <geom name="fl_lower" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="yellow"/>
          <site name="fl_foot" pos="0 0 -0.12" size="0.05" type="sphere"/>
          <geom name="fl_foot" type="sphere" pos="0 0 -0.12" size="0.05" material="red"/>
        </body>
      </body>
      
      <!-- Front Right Leg -->
      <body name="fr_upper" pos="-0.25 0.15 -0.05">
        <joint name="fr_hip_x" type="hinge" axis="1 0 0" range="-60 0" damping="3"/>
        <joint name="fr_hip_y" type="hinge" axis="0 1 0" range="-20 20" damping="3"/>
        <geom name="fr_upper" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="orange"/>
        
        <body name="fr_lower" pos="0 0 -0.12">
          <joint name="fr_knee" type="hinge" axis="1 0 0" range="0 150" damping="2"/>
          <geom name="fr_lower" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="yellow"/>
          <site name="fr_foot" pos="0 0 -0.12" size="0.05" type="sphere"/>
          <geom name="fr_foot" type="sphere" pos="0 0 -0.12" size="0.05" material="red"/>
        </body>
      </body>
      
      <!-- Back Left Leg -->
      <body name="bl_upper" pos="0.25 -0.15 -0.05">
        <joint name="bl_hip_x" type="hinge" axis="1 0 0" range="-60 0" damping="3"/>
        <joint name="bl_hip_y" type="hinge" axis="0 1 0" range="-20 20" damping="3"/>
        <geom name="bl_upper" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="orange"/>
        
        <body name="bl_lower" pos="0 0 -0.12">
          <joint name="bl_knee" type="hinge" axis="1 0 0" range="0 150" damping="2"/>
          <geom name="bl_lower" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="yellow"/>
          <site name="bl_foot" pos="0 0 -0.12" size="0.05" type="sphere"/>
          <geom name="bl_foot" type="sphere" pos="0 0 -0.12" size="0.05" material="red"/>
        </body>
      </body>
      
      <!-- Back Right Leg -->
      <body name="br_upper" pos="-0.25 -0.15 -0.05">
        <joint name="br_hip_x" type="hinge" axis="1 0 0" range="-60 0" damping="3"/>
        <joint name="br_hip_y" type="hinge" axis="0 1 0" range="-20 20" damping="3"/>
        <geom name="br_upper" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="orange"/>
        
        <body name="br_lower" pos="0 0 -0.12">
          <joint name="br_knee" type="hinge" axis="1 0 0" range="0 150" damping="2"/>
          <geom name="br_lower" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.12" material="yellow"/>
          <site name="br_foot" pos="0 0 -0.12" size="0.05" type="sphere"/>
          <geom name="br_foot" type="sphere" pos="0 0 -0.12" size="0.05" material="red"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <!-- Actuators matching Unity's action space exactly -->
  <actuator>
    <!-- Upper leg actuators (2 per leg for X and Y) -->
    <position name="fl_hip_x_motor" joint="fl_hip_x" kp="40" forcerange="-150 150"/>
    <position name="fl_hip_y_motor" joint="fl_hip_y" kp="40" forcerange="-150 150"/>
    <position name="fr_hip_x_motor" joint="fr_hip_x" kp="40" forcerange="-150 150"/>
    <position name="fr_hip_y_motor" joint="fr_hip_y" kp="40" forcerange="-150 150"/>
    <position name="bl_hip_x_motor" joint="bl_hip_x" kp="40" forcerange="-150 150"/>
    <position name="bl_hip_y_motor" joint="bl_hip_y" kp="40" forcerange="-150 150"/>
    <position name="br_hip_x_motor" joint="br_hip_x" kp="40" forcerange="-150 150"/>
    <position name="br_hip_y_motor" joint="br_hip_y" kp="40" forcerange="-150 150"/>
    
    <!-- Lower leg actuators (1 per leg) -->
    <position name="fl_knee_motor" joint="fl_knee" kp="40" forcerange="-150 150"/>
    <position name="fr_knee_motor" joint="fr_knee" kp="40" forcerange="-150 150"/>
    <position name="bl_knee_motor" joint="bl_knee" kp="40" forcerange="-150 150"/>
    <position name="br_knee_motor" joint="br_knee" kp="40" forcerange="-150 150"/>
  </actuator>
  
  <!-- Sensors for ground contact -->
  <sensor>
    <touch name="fl_foot_touch" site="fl_foot"/>
    <touch name="fr_foot_touch" site="fr_foot"/>
    <touch name="bl_foot_touch" site="bl_foot"/>
    <touch name="br_foot_touch" site="br_foot"/>
  </sensor>
</mujoco>
"""

# -----------------------------------------------------------------------------------
# Environment definition
# -----------------------------------------------------------------------------------

OBS_SIZE = 172
ACTION_SIZE = 12
MAX_EPISODE_STEPS = 500
TARGET_SPEED = 1.0  # m/s


class CrawlerEnv:
    def __init__(self):
        # Create temporary file for XML
        self.xml_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        self.xml_file.write(CRAWLER_XML)
        self.xml_file.close()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_file.name)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and actuator indices for easier access
        self.joint_names = [
            'fl_hip_x', 'fl_hip_y', 'fr_hip_x', 'fr_hip_y',
            'bl_hip_x', 'bl_hip_y', 'br_hip_x', 'br_hip_y',
            'fl_knee', 'fr_knee', 'bl_knee', 'br_knee'
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        # Body IDs for observations
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        # Foot site IDs for ground contact
        self.foot_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot")
            for leg in ['fl', 'fr', 'bl', 'br']
        ]
        
        # Orientation cube state (simulated)
        self.target_pos = np.array([10.0, 0.0, 0.25])  # Match new torso height
        self.orientation_forward = np.array([1.0, 0.0, 0.0])
        
        self.step_counter = 0
        self.reset()
    
    def __del__(self):
        if hasattr(self, 'xml_file'):
            os.unlink(self.xml_file.name)
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # Random starting rotation
        start_yaw = np.random.uniform(0, 2 * math.pi)
        quat = np.array([np.cos(start_yaw/2), 0, 0, np.sin(start_yaw/2)])
        
        # Set initial position and orientation (qpos for freejoint: x,y,z,qw,qx,qy,qz)
        self.data.qpos[0:3] = [0, 0, 0.25]  # position
        self.data.qpos[3:7] = [quat[0], quat[1], quat[2], quat[3]]  # quaternion (w,x,y,z)
        
        # Set initial joint positions for stable stance with shorter legs
        joint_init = [
            -0.3, 0.0,  # fl_hip_x, fl_hip_y (less bent for shorter legs)
            -0.3, 0.0,  # fr_hip_x, fr_hip_y
            -0.3, 0.0,  # bl_hip_x, bl_hip_y
            -0.3, 0.0,  # br_hip_x, br_hip_y
            0.6, 0.6, 0.6, 0.6  # knees bent appropriately for new leg length
        ]
        
        for i, joint_id in enumerate(self.joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = np.radians(joint_init[i])
        
        # Forward dynamics to settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        self.step_counter = 0
        self._update_orientation_cube()
        
        # Initialize distance tracking for reward calculation
        self._last_distance = np.linalg.norm(self.target_pos - self.data.xpos[self.torso_id])
        
        return self._get_obs()
    
    def _update_orientation_cube(self):
        """Simulates Unity's OrientationCubeController"""
        torso_pos = self.data.xpos[self.torso_id]
        
        # Direction from agent to target (flattened on Z)
        dir_to_target = self.target_pos - torso_pos
        dir_to_target[2] = 0  # Flatten on Z
        
        # Always point toward the target (even if close)
        distance_to_target = np.linalg.norm(dir_to_target)
        if distance_to_target > 0.1:  # Update direction if not very close
            self.orientation_forward = dir_to_target / distance_to_target
        
        # If we're close to target, move the target further away
        if distance_to_target < 2.0:
            # Move target further in the same direction
            current_direction = self.orientation_forward
            self.target_pos = torso_pos + current_direction * 10.0
            self.target_pos[2] = 0.25  # Keep target at torso height
    
    def _get_obs(self):
        """Generate observation matching Unity's format"""
        obs = []
        
        # Get torso info
        torso_pos = self.data.xpos[self.torso_id]
        torso_quat = self.data.xquat[self.torso_id]
        
        # Get torso velocity properly (6-vector per body: linear(3) | angular(3))
        torso_vel = self.data.cvel[self.torso_id][:3]  # linear velocity
        # angular velocity not used but available via self.data.cvel[self.torso_id][3:]
        
        # Get body forward direction (rotation matrix from quaternion)
        torso_xmat = self.data.xmat[self.torso_id * 9: (self.torso_id + 1) * 9].reshape(3, 3)
        body_forward = torso_xmat[:, 0]  # X axis is forward
        
        # Velocity matching observations
        vel_goal = self.orientation_forward * TARGET_SPEED
        avg_vel = torso_vel
        
        # 1. Distance between velocities
        obs.append(np.linalg.norm(vel_goal - avg_vel))
        
        # 2. Average velocity relative to orientation cube
        obs.extend(avg_vel.tolist())
        
        # 3. Goal velocity relative to orientation cube
        obs.extend(vel_goal.tolist())
        
        # 4. Rotation delta (quaternion components)
        dot = np.dot(body_forward, self.orientation_forward)
        cross = np.cross(body_forward, self.orientation_forward)
        obs.append(dot)
        obs.extend(cross.tolist())
        
        # 5. Target position relative to agent
        target_rel = self.target_pos - torso_pos
        obs.extend(target_rel.tolist())
        
        # 6. Ground distance (simplified)
        obs.append(torso_pos[2] / 10.0)
        
        # 7. Body part observations (matching Unity's CollectObservationBodyPart)
        for i, joint_id in enumerate(self.joint_ids):
            # Ground contact (check foot sensors for lower legs)
            if i >= 8:  # Lower leg joints
                foot_idx = i - 8
                contact = self.data.sensordata[foot_idx] > 0.01  # Touch sensor threshold
                obs.append(1.0 if contact else 0.0)
            else:
                obs.append(0.0)  # Upper legs don't have ground contact
            
            # Joint strength (normalized) - we'll use a fixed value for now
            obs.append(0.5)
            
            # Joint position and velocity
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            
            obs.append(self.data.qpos[qpos_addr])  # position
            obs.append(self.data.qvel[dof_addr])   # velocity
        
        # 8. CLEAR DIRECTIONAL GUIDANCE (relative to agent's body orientation)
        # Direction to target in agent's local coordinate system
        target_dir_world = self.orientation_forward  # World direction to target
        
        # Transform to agent's local coordinates using rotation matrix
        target_dir_local = np.dot(torso_xmat.T, target_dir_world)  # Transform to body coordinates
        obs.extend(target_dir_local.tolist())  # [forward, right, up] relative to agent
        
        # How much should agent turn? (left=-1, straight=0, right=+1)
        turn_amount = np.arctan2(target_dir_local[1], target_dir_local[0]) / np.pi  # Normalize to [-1,1]
        obs.append(turn_amount)
        
        # Distance to target (normalized)
        distance_to_target = np.linalg.norm(self.target_pos - torso_pos)
        obs.append(min(distance_to_target / 10.0, 1.0))  # Normalize, cap at 1.0
        
        # Pad to OBS_SIZE
        while len(obs) < OBS_SIZE:
            obs.append(0.0)
        
        return np.array(obs[:OBS_SIZE], dtype=np.float32)
    
    def step(self, action: np.ndarray):
        # Action mapping (12 values):
        # 0-7  – Upper-leg joint targets (X then Y for each leg in order FL, FR, BL, BR)
        # 8-11 – Knee joint targets (one DOF per leg in same order)
        
        # Apply target angles to position actuators
        for i in range(12):
            joint_id = self.joint_ids[i]
            joint_range = self.model.jnt_range[joint_id]
            norm = (action[i] + 1.0) * 0.5  # [-1,1] → [0,1]
            target_angle = joint_range[0] + norm * (joint_range[1] - joint_range[0])
            self.data.ctrl[i] = target_angle  # position actuator takes angle in radians
        
        mujoco.mj_step(self.model, self.data)
        self.step_counter += 1
        
        # Update orientation helper
        self._update_orientation_cube()
        
        # Reward components (enhanced for better goal-seeking) ----------------
        torso_pos = self.data.xpos[self.torso_id]
        torso_vel = self.data.cvel[self.torso_id][:3]
        torso_xmat = self.data.xmat[self.torso_id * 9:(self.torso_id + 1) * 9].reshape(3, 3)
        body_forward = torso_xmat[:, 0]
        
        vel_goal = self.orientation_forward * TARGET_SPEED
        vel_delta = np.clip(np.linalg.norm(torso_vel - vel_goal), 0, TARGET_SPEED)
        match_speed_reward = (1 - (vel_delta / TARGET_SPEED) ** 2) ** 2
        look_at_reward = (np.dot(self.orientation_forward, body_forward) + 1) * 0.5
        
        # STRONG directional movement rewards
        # 1. Reward velocity in the correct direction (dot product)
        forward_velocity_reward = np.dot(torso_vel, self.orientation_forward)
        forward_velocity_reward = max(0, forward_velocity_reward)  # Only positive movement
        
        # 2. Penalty for moving in wrong direction
        wrong_direction_penalty = min(0, np.dot(torso_vel, self.orientation_forward))
        
        # 3. Strong heading alignment reward
        heading_alignment = np.dot(self.orientation_forward, body_forward)  # -1 to 1
        heading_reward = max(0, heading_alignment) ** 2  # Square to make it stronger
        
        # 4. CRITICAL STABILITY REWARDS - keep crawler upright!
        # Check if torso is upright (Z-axis should point up)
        up_vector = np.array([0, 0, 1])
        torso_up = torso_xmat[:, 2]  # Z-axis of torso orientation
        upright_reward = np.dot(torso_up, up_vector)  # 1.0 = perfectly upright, -1.0 = upside down
        upright_reward = max(0, upright_reward) ** 2  # Square for stronger effect, only positive
        
        # Strong penalty for being upside down or on side
        if upright_reward < 0.3:  # If significantly tilted
            stability_penalty = -0.5  # Large penalty
        else:
            stability_penalty = 0.0
        
        # Height reward - encourage staying at proper height
        target_height = 0.25
        height_diff = abs(torso_pos[2] - target_height)
        height_reward = max(0, 1.0 - height_diff * 2.0)  # Penalty for being too high or low
        
        # Combined reward with stability as priority
        base_reward = match_speed_reward * look_at_reward
        stability_component = upright_reward * 0.4 + height_reward * 0.1 + stability_penalty
        directional_component = forward_velocity_reward * 0.2 + wrong_direction_penalty * 0.05 + heading_reward * 0.1
        
        reward = base_reward + stability_component + directional_component
        
        # Small alive bonus (only if upright)
        if upright_reward > 0.5:  # Only if reasonably upright
            reward += 0.02
        
        # Episode termination conditions
        too_low = torso_pos[2] < 0.1  # Fallen down
        too_tilted = upright_reward < 0.2  # Severely tilted/upside down
        max_steps = self.step_counter >= MAX_EPISODE_STEPS
        
        done = max_steps or too_low or too_tilted
        return self._get_obs(), reward, done
    
    def get_state_for_viz(self):
        """Get state for visualization"""
        torso_pos = self.data.xpos[self.torso_id]
        torso_quat = self.data.xquat[self.torso_id]
        
        # Get ALL joint angles for visualization (12 joints total)
        joint_angles = []
        for joint_name in self.joint_names:  # Use all 12 joints
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            joint_angles.append(float(self.data.qpos[qpos_addr]))
        
        # Add debug info about target direction
        target_distance = np.linalg.norm(self.target_pos - torso_pos)
        
        # Add directional debug info
        torso_vel = self.data.cvel[self.torso_id][:3]
        forward_velocity = np.dot(torso_vel, self.orientation_forward)
        torso_xmat = self.data.xmat[self.torso_id * 9:(self.torso_id + 1) * 9].reshape(3, 3)
        body_forward = torso_xmat[:, 0]
        heading_alignment = np.dot(self.orientation_forward, body_forward)
        
        # Add stability debug info
        up_vector = np.array([0, 0, 1])
        torso_up = torso_xmat[:, 2]
        upright_score = np.dot(torso_up, up_vector)
        
        return {
            "basePos": torso_pos.tolist(),
            "baseOri": torso_quat.tolist(),
            "jointAngles": joint_angles,
            "targetPos": self.target_pos.tolist(),
            "targetDistance": float(target_distance),
            "orientationForward": self.orientation_forward.tolist(),
            "forwardVelocity": float(forward_velocity),
            "headingAlignment": float(heading_alignment),
            "bodyForward": body_forward.tolist(),
            "uprightScore": float(upright_score),
            "torsoHeight": float(torso_pos[2])
        }


# -----------------------------------------------------------------------------------
# PPO agent (same as before, no changes needed)
# -----------------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(OBS_SIZE, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.actor_mean = nn.Linear(128, ACTION_SIZE)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -0.8))  # Even higher exploration for directional learning
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):  # type: ignore[override]
        h = self.shared(obs)
        return self.actor_mean(h), self.log_std.expand_as(self.actor_mean(h)), self.critic(h)


POLICIES_DIR = "policies"
BATCH_SIZE = 2048
MINI_BATCH = 256
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.001
LR = 3e-4


def _export_model_onnx(model: nn.Module, path: str):
    dummy = torch.zeros((1, OBS_SIZE), dtype=torch.float32)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})


async def train_crawler(websocket: WebSocket):
    os.makedirs(POLICIES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]
    model_filename = f"crawler_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)

    envs: List[CrawlerEnv] = [CrawlerEnv() for _ in range(8)]
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    obs = torch.tensor([e.reset() for e in envs], dtype=torch.float32)
    ep_counter = 0
    step_buffer: list[dict] = []
    while ep_counter < 2000:  # episodes
        # collect rollout
        with torch.no_grad():
            mean, log_std, value = model(obs)
            std = log_std.exp()
            actions = mean + std * torch.randn_like(mean)
            probs = (-0.5 * ((actions - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        actions_np = actions.clamp(-1, 1).cpu().numpy()
        step_obs = []
        rewards = []
        dones = []
        for idx, env in enumerate(envs):
            nobs, rew, dn = env.step(actions_np[idx])
            step_obs.append(nobs)
            rewards.append(rew)
            dones.append(dn)
        next_obs = torch.tensor(step_obs, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        step_buffer.append({"obs": obs, "actions": actions, "logp": probs, "reward": rewards_t, "done": dones_t, "value": value})
        obs = next_obs

        # stream first env every 4 steps for clearer observation
        if len(step_buffer) % TRAINING_STREAM_FREQ == 0:
            e0 = envs[0]
            state = e0.get_state_for_viz()
            await websocket.send_json({
                "type": "train_step",
                "state": state,
                "episode": ep_counter + 1,
            })
            # Add delay to slow down training visualization (optional)
            if ENABLE_TRAINING_DELAYS:
                await asyncio.sleep(TRAINING_DELAY)

        # finish episode if any env done
        for i, dn in enumerate(dones):
            if dn:
                ep_counter += 1
                envs[i].reset()

        # once buffer big enough, update policy
        if len(step_buffer) * len(envs) >= BATCH_SIZE:
            # convert buffer to tensors
            obs_cat = torch.cat([b["obs"] for b in step_buffer])
            act_cat = torch.cat([b["actions"] for b in step_buffer])
            logp_cat = torch.cat([b["logp"] for b in step_buffer])
            rew_cat = torch.cat([b["reward"] for b in step_buffer])
            done_cat = torch.cat([b["done"] for b in step_buffer])
            val_cat = torch.cat([b["value"] for b in step_buffer])
            # compute advantages via GAE
            adv = torch.zeros_like(rew_cat)
            gae = 0.0
            for t in reversed(range(rew_cat.shape[0])):
                delta = rew_cat[t] + GAMMA * (1.0 - done_cat[t]) * val_cat[t] - val_cat[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1.0 - done_cat[t]) * gae
                adv[t] = gae
            returns = adv + val_cat
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # PPO update
            for _ in range(EPOCHS):
                idx = torch.randperm(obs_cat.shape[0])
                for start in range(0, obs_cat.shape[0], MINI_BATCH):
                    mb_idx = idx[start:start + MINI_BATCH]
                    mb_obs = obs_cat[mb_idx]
                    mb_act = act_cat[mb_idx]
                    mb_logp_old = logp_cat[mb_idx]
                    mb_adv = adv[mb_idx]
                    mb_ret = returns[mb_idx]
                    mean, log_std, value = model(mb_obs)
                    std = log_std.exp()
                    logp = (-0.5 * ((mb_act - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                    ratio = (logp - mb_logp_old).exp()
                    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(ratio * mb_adv, clip_adv).mean() - ENT_COEF * (-logp.mean())
                    value_loss = ((mb_ret - value) ** 2).mean()
                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # gather simple metrics for frontend
            avg_reward = float(rew_cat.mean().cpu().item())
            avg_loss = float(loss.detach().cpu().item())
            step_buffer = []
            # send progress every update (include reward/loss)
            await websocket.send_json({"type": "progress", "episode": ep_counter + 1, "reward": avg_reward, "loss": avg_loss})

    _export_model_onnx(model, model_path)
    await websocket.send_json({"type": "trained", "file_url": f"/policies/{model_filename}", "model_filename": model_filename, "timestamp": timestamp, "session_uuid": session_uuid})


# -----------------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------------

_ORT_CACHE: dict[str, "onnxruntime.InferenceSession"] = {}


def infer_action_crawler(obs: List[float], model_filename: str | None = None):
    import onnxruntime as ort
    if model_filename is None:
        files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("crawler_policy_") and f.endswith(".onnx")]
        files.sort(reverse=True)
        model_filename = files[0]
    if model_filename not in _ORT_CACHE:
        sess = ort.InferenceSession(os.path.join(POLICIES_DIR, model_filename), providers=["CPUExecutionProvider"])
        _ORT_CACHE[model_filename] = sess
    # pad or truncate to OBS_SIZE so it matches network input
    if len(obs) < OBS_SIZE:
        obs_adj = obs + [0.0] * (OBS_SIZE - len(obs))
    else:
        obs_adj = obs[:OBS_SIZE]
    inp = np.array([obs_adj], dtype=np.float32)
    out = _ORT_CACHE[model_filename].run(None, {"input": inp})[0]
    return out[0].tolist()  # return continuous 12-D vector


async def run_crawler(websocket: WebSocket, model_filename: str | None = None):
    """Stream live inference rollout of a single CrawlerEnv to the websocket."""
    env = CrawlerEnv()
    episode = 0
    obs = env.reset()
    from starlette.websockets import WebSocketState
    while websocket.application_state == WebSocketState.CONNECTED:
        act_vec = infer_action_crawler(obs.tolist(), model_filename)
        nobs, _, done = env.step(np.array(act_vec, dtype=np.float32))
        
        state = env.get_state_for_viz()
        await websocket.send_json({
            "type": "run_step",
            "state": state,
            "episode": episode + 1,
        })
        
        # Slow down inference for better observation
        await asyncio.sleep(INFERENCE_DELAY)  # 30ms delay for smooth but visible movement
        
        if done:
            episode += 1
            obs = env.reset()
        else:
            obs = nobs
        await asyncio.sleep(0)  # cooperative yield 