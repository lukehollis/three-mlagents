import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket

# Environment constants
SMALL_GOAL = 7
LARGE_GOAL = 17
MIN_POS = 0
MAX_POS = 20
START_POS = 10

# Path where the trained policy will be exported (same as in main.py)
MODEL_PATH = "policies/basic_policy.onnx"

# --- Neural-net definition ----------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

# ----------------------------------------------------------------------------

async def train_basic(websocket: WebSocket):
    """Train a simple Q-learning agent and stream progress via websocket."""
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    gamma = 0.9
    epsilon = 1.0
    episodes = 300

    for ep in range(episodes):
        pos = START_POS
        total_reward = 0.0
        episode_loss = 0.0
        step_count = 0
        steps = 0
        while True:
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 3)  # explore
            else:
                with torch.no_grad():
                    qvals = net(torch.tensor([[pos]], dtype=torch.float32))
                    action = int(torch.argmax(qvals).item())

            # Map action index to environment delta
            delta = [-1, 0, 1][action]

            next_pos = pos + delta
            next_pos = max(MIN_POS, min(MAX_POS, next_pos))

            reward = -0.01
            done = False
            if next_pos == SMALL_GOAL:
                reward += 0.1
                done = True
            if next_pos == LARGE_GOAL:
                reward += 1.0
                done = True

            total_reward += reward

            # Q-learning update
            q_pred = net(torch.tensor([[pos]], dtype=torch.float32))[0, action]
            with torch.no_grad():
                q_next_max = net(torch.tensor([[next_pos]], dtype=torch.float32)).max()
                q_target = reward + (0.0 if done else gamma * q_next_max.item())
                q_target = torch.tensor(q_target)
            loss = (q_pred - q_target) ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()
            step_count += 1

            # Stream every step for visualization
            await websocket.send_json({
                "type": "train_step",
                "pos": next_pos,
                "episode": ep + 1,
                "step": steps,
                "done": done,
            })

            await asyncio.sleep(0.02)  # allow UI to update

            pos = next_pos
            steps += 1
            if done or steps >= 50:
                break

        # Anneal exploration
        epsilon = max(0.05, epsilon * 0.995)

        if (ep + 1) % 20 == 0:
            avg_loss = episode_loss / max(1, step_count)
            await websocket.send_json({
                "type": "progress",
                "episode": ep + 1,
                "reward": round(total_reward, 3),
                "loss": round(avg_loss, 5),
            })

    # Export policy to ONNX
    dummy_input = torch.tensor([[START_POS]], dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy_input,
        MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    await websocket.send_json({
        "type": "trained",
        "file_url": f"/policies/{os.path.basename(MODEL_PATH)}",
    })

# -----------------------------------------------------------------------------
# Inference helper -------------------------------------------------------------

_ort_session = None  # lazy initialisation

async def infer_action(position: int):
    """Run the trained ONNX policy and return an action index (0,1,2)."""
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort
        _ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    inp = np.array([[position]], dtype=np.float32)
    outputs = _ort_session.run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 