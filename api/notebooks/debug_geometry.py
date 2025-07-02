#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("Debugging crawler geometry...")
env = CrawlerEnv()
obs = env.reset()

# Check initial setup
torso_pos = env.data.xpos[env.torso_id]
print(f"Torso position: {torso_pos}")

# Get foot positions  
for i, leg in enumerate(['fl', 'fr', 'bl', 'br']):
    foot_site_id = env.foot_sites[i]
    foot_pos = env.data.site_xpos[foot_site_id]
    print(f"{leg} foot position: {foot_pos}")
    
print(f"Ground plane is at Z=0")
print(f"Foot-to-ground distances:")
for i, leg in enumerate(['fl', 'fr', 'bl', 'br']):
    foot_site_id = env.foot_sites[i]  
    foot_z = env.data.site_xpos[foot_site_id][2]
    print(f"  {leg}: {foot_z:.3f}m {'(BELOW GROUND!)' if foot_z < 0 else ''}")

# Check joint ranges
print(f"\nJoint ranges:")
for joint_name in env.joint_names:
    joint_id = env.joint_ids[env.joint_names.index(joint_name)]
    joint_range = env.model.jnt_range[joint_id]
    print(f"  {joint_name}: [{joint_range[0]:.1f}, {joint_range[1]:.1f}] degrees") 