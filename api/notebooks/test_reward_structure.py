#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("=== TESTING REWARD STRUCTURE ===")

def test_action_sequence(env, actions, name):
    print(f"\n--- {name} ---")
    env.reset()
    
    total_reward = 0
    for i, action in enumerate(actions):
        obs, reward, done = env.step(action.astype(np.float32))
        total_reward += reward
        
        if i % 10 == 0 or i < 5:
            torso_pos = env.data.xpos[env.torso_id]
            torso_vel = env.data.cvel[env.torso_id][:3]
            torso_xmat = env.data.xmat[env.torso_id * 9:(env.torso_id + 1) * 9].reshape(3, 3)
            body_forward = torso_xmat[:, 0]
            
            vel_goal = env.orientation_forward * 1.0
            vel_delta = np.clip(np.linalg.norm(torso_vel - vel_goal), 0, 1.0)
            speed_reward = (1 - (vel_delta / 1.0) ** 2) ** 2
            look_reward = (np.dot(env.orientation_forward, body_forward) + 1) * 0.5
            base_reward = speed_reward * look_reward
            
            print(f"  Step {i:2d}: pos={torso_pos[0]:5.2f}, vel={np.linalg.norm(torso_vel):5.2f}, "
                  f"speed_r={speed_reward:.3f}, look_r={look_reward:.3f}, total_r={reward:.3f}")
    
    print(f"  Final total reward: {total_reward:.2f}")
    return total_reward

env = CrawlerEnv()

# Test 1: Zero actions (baseline)
actions_zero = [np.zeros(12) for _ in range(50)]
test_action_sequence(env, actions_zero, "Zero Actions")

# Test 2: Try to move forward by cycling legs
forward_cycle = []
for t in range(50):
    # Simple alternating gait attempt
    phase = t * 0.2
    action = np.array([
        -0.3 + 0.2 * np.sin(phase),      # fl_hip_x
        0.1 * np.sin(phase + np.pi),     # fl_hip_y
        -0.3 + 0.2 * np.sin(phase + np.pi), # fr_hip_x  
        0.1 * np.sin(phase),             # fr_hip_y
        -0.3 + 0.2 * np.sin(phase + np.pi), # bl_hip_x
        0.1 * np.sin(phase),             # bl_hip_y
        -0.3 + 0.2 * np.sin(phase),      # br_hip_x
        0.1 * np.sin(phase + np.pi),     # br_hip_y
        0.2 + 0.1 * np.sin(phase * 2),   # fl_knee
        0.2 + 0.1 * np.sin(phase * 2 + np.pi), # fr_knee
        0.2 + 0.1 * np.sin(phase * 2 + np.pi), # bl_knee  
        0.2 + 0.1 * np.sin(phase * 2),   # br_knee
    ])
    forward_cycle.append(action)

test_action_sequence(env, forward_cycle, "Forward Cycling Gait")

# Test 3: Strong forward push
forward_push = [np.array([-0.8, 0, -0.8, 0, -0.8, 0, -0.8, 0, 0.1, 0.1, 0.1, 0.1]) for _ in range(50)]
test_action_sequence(env, forward_push, "Strong Forward Push")

print("\n=== ANALYSIS ===")
print("If all rewards are ~0.05, the reward structure is broken.")
print("We should see higher rewards when the agent moves forward at ~1 m/s.") 