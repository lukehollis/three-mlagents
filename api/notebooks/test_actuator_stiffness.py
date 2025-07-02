#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("=== TESTING ACTUATOR STIFFNESS ===")

def test_leg_coordination(env, kp_value, name):
    print(f"\n--- {name} (kp={kp_value}) ---")
    
    # Manually set actuator stiffness
    for i in range(12):
        env.model.actuator_gainprm[i][0] = kp_value
    
    env.reset()
    
    # Test alternating gait pattern
    for step in range(30):
        # Simple trot gait: FL+BR together, FR+BL together
        phase = step * 0.3
        action = np.array([
            -0.4 + 0.3 * np.sin(phase),      # fl_hip_x (lift)
            0.1 * np.sin(phase),             # fl_hip_y (side)
            -0.4 + 0.3 * np.sin(phase + np.pi), # fr_hip_x (opposite)
            0.1 * np.sin(phase + np.pi),     # fr_hip_y
            -0.4 + 0.3 * np.sin(phase + np.pi), # bl_hip_x (opposite) 
            0.1 * np.sin(phase + np.pi),     # bl_hip_y
            -0.4 + 0.3 * np.sin(phase),      # br_hip_x (together with FL)
            0.1 * np.sin(phase),             # br_hip_y
            0.2 + 0.2 * np.sin(phase * 2),   # fl_knee
            0.2 + 0.2 * np.sin(phase * 2 + np.pi), # fr_knee
            0.2 + 0.2 * np.sin(phase * 2 + np.pi), # bl_knee
            0.2 + 0.2 * np.sin(phase * 2),   # br_knee
        ], dtype=np.float32)
        
        obs, reward, done = env.step(action)
        
        if step % 10 == 0:
            # Check individual joint positions
            joint_positions = []
            for i, joint_name in enumerate(env.joint_names):
                joint_id = env.joint_ids[i]
                qpos_addr = env.model.jnt_qposadr[joint_id]
                joint_positions.append(env.data.qpos[qpos_addr])
            
            torso_vel = env.data.cvel[env.torso_id][:3]
            print(f"  Step {step:2d}: vel={np.linalg.norm(torso_vel):.2f}, reward={reward:.3f}")
            print(f"    FL hip: {np.degrees(joint_positions[0]):6.1f}°, FR hip: {np.degrees(joint_positions[2]):6.1f}°")
            print(f"    BL hip: {np.degrees(joint_positions[4]):6.1f}°, BR hip: {np.degrees(joint_positions[6]):6.1f}°")
            print(f"    Knees: FL={np.degrees(joint_positions[8]):5.1f}°, FR={np.degrees(joint_positions[9]):5.1f}°, BL={np.degrees(joint_positions[10]):5.1f}°, BR={np.degrees(joint_positions[11]):5.1f}°")

# Test different stiffness values
env = CrawlerEnv()

test_leg_coordination(env, 120, "Current (Very Stiff)")
test_leg_coordination(env, 60, "Medium Stiff") 
test_leg_coordination(env, 30, "Soft")
test_leg_coordination(env, 15, "Very Soft")

print("\n=== ANALYSIS ===")
print("Look for:")
print("- Individual joint angles varying independently")
print("- Smooth transitions between positions") 
print("- Natural trot-like coordination (FL+BR vs FR+BL)")
print("- Less synchronized 'snapping' to targets") 