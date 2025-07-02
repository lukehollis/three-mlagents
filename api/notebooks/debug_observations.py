#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("=== DEBUGGING OBSERVATIONS & REWARDS ===")
env = CrawlerEnv()
obs = env.reset()

print(f"Observation space size: {len(obs)}")
print(f"Target position: {env.target_pos}")
print(f"Orientation forward: {env.orientation_forward}")

# Test a few steps with different actions
actions_to_test = [
    ("Zero", np.zeros(12)),
    ("Random", np.random.uniform(-1, 1, 12)),
    ("Forward-biased", np.array([-0.5, 0, -0.5, 0, -0.5, 0, -0.5, 0, 0.2, 0.2, 0.2, 0.2])),
]

for name, action in actions_to_test:
    print(f"\n--- Testing {name} action ---")
    env.reset()
    
    for step in range(5):
        obs, reward, done = env.step(action.astype(np.float32))
        
        # Break down the reward
        torso_pos = env.data.xpos[env.torso_id]
        torso_vel = env.data.cvel[env.torso_id][:3]
        torso_xmat = env.data.xmat[env.torso_id * 9:(env.torso_id + 1) * 9].reshape(3, 3)
        body_forward = torso_xmat[:, 0]
        
        vel_goal = env.orientation_forward * 1.0  # TARGET_SPEED = 1.0
        vel_delta = np.clip(np.linalg.norm(torso_vel - vel_goal), 0, 1.0)
        match_speed_reward = (1 - (vel_delta / 1.0) ** 2) ** 2
        look_at_reward = (np.dot(env.orientation_forward, body_forward) + 1) * 0.5
        
        print(f"  Step {step}:")
        print(f"    Torso pos: {torso_pos}")
        print(f"    Torso vel: {torso_vel} (speed: {np.linalg.norm(torso_vel):.3f})")
        print(f"    Body forward: {body_forward}")
        print(f"    Target vel: {vel_goal}")
        print(f"    Speed reward: {match_speed_reward:.3f}")
        print(f"    Look reward: {look_at_reward:.3f}")
        print(f"    Total reward: {reward:.3f}")
        
        if step == 0:
            # Check key observation components
            print(f"    Key obs[0] (vel distance): {obs[0]:.3f}")
            print(f"    Obs[1:4] (avg vel): {obs[1:4]}")
            print(f"    Obs[4:7] (goal vel): {obs[4:7]}")
            print(f"    Ground distance obs: {obs[13]:.3f}")

print(f"\n=== TESTING TARGET UPDATES ===")
env.reset()
for i in range(20):
    obs, reward, done = env.step(np.zeros(12, dtype=np.float32))
    if i % 5 == 0:
        print(f"Step {i}: target={env.target_pos}, forward={env.orientation_forward}") 