#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("Testing stable stance...")
env = CrawlerEnv()
obs = env.reset()

print(f"Initial torso Z: {env.data.xpos[env.torso_id][2]:.3f}")

# Actions to extend legs into stable stance
# Hip X: more negative = legs extend down more
# Hip Y: 0 = no side bend  
# Knees: less positive = legs extend more
stable_action = np.array([
    -0.3, 0.0,  # fl_hip_x, fl_hip_y - extend front left leg
    -0.3, 0.0,  # fr_hip_x, fr_hip_y - extend front right leg  
    -0.3, 0.0,  # bl_hip_x, bl_hip_y - extend back left leg
    -0.3, 0.0,  # br_hip_x, br_hip_y - extend back right leg
    0.1, 0.1, 0.1, 0.1  # slightly bent knees for stability
], dtype=np.float32)

# Apply stable stance and let it settle
for i in range(300):
    obs, rew, done = env.step(stable_action)
    if i % 50 == 0:
        torso_z = env.data.xpos[env.torso_id][2]
        print(f"Step {i}: torso Z = {torso_z:.3f}")
        
        # Check foot positions
        for j, leg in enumerate(['fl', 'fr', 'bl', 'br']):
            foot_z = env.data.site_xpos[env.foot_sites[j]][2]
            contact = env.data.sensordata[j] > 0.01
            print(f"  {leg} foot: Z={foot_z:.3f}, contact={contact}")

print(f"Final torso Z: {env.data.xpos[env.torso_id][2]:.3f}")
print("If this settles around 0.2-0.25m with foot contacts, physics is working!") 