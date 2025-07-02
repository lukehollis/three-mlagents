#!/usr/bin/env python3

from examples.crawler import CrawlerEnv
import numpy as np

print("Testing crawler physics...")
env = CrawlerEnv()
obs = env.reset()

print(f"Initial torso Z: {env.data.xpos[env.torso_id][2]:.3f}")

# zero action for 300 sim steps
for i in range(300):
    obs, rew, done = env.step(np.zeros(12, dtype=np.float32))
    if i % 50 == 0:
        print(f"Step {i}: torso Z = {env.data.xpos[env.torso_id][2]:.3f}")

print(f"Final torso Z after 300 steps: {env.data.xpos[env.torso_id][2]:.3f}")

if env.data.xpos[env.torso_id][2] > 0.35:
    print("❌ Problem: Torso didn't fall! Actuators are too stiff.")
elif env.data.xpos[env.torso_id][2] < 0.15:
    print("❌ Problem: Torso fell through ground or collapsed completely.")
else:
    print("✅ Good: Torso settled to reasonable height under gravity.") 