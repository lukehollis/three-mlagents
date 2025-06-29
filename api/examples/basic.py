import os
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import WebSocket
from datetime import datetime
import uuid

# Environment constants
SMALL_GOAL = 7
LARGE_GOAL = 17
MIN_POS = 0
MAX_POS = 20
START_POS = 10

# Base path for policies directory
POLICIES_DIR = "policies"

# --- Neural-net definition ----------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Use one-hot encoding like Unity (21 positions: 0-20)
        self.fc1 = nn.Linear(21, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

# ----------------------------------------------------------------------------

def position_to_onehot(pos):
    """Convert position to one-hot encoding like Unity implementation."""
    onehot = np.zeros(21, dtype=np.float32)  # positions 0-20
    onehot[pos] = 1.0
    return onehot

async def train_basic(websocket: WebSocket):
    """Train a simple Q-learning agent and stream progress via websocket."""
    # Generate timestamped filename with UUID for guaranteed uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_uuid = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
    model_filename = f"basic_policy_{timestamp}_{session_uuid}.onnx"
    model_path = os.path.join(POLICIES_DIR, model_filename)
    
    # Ensure policies directory exists
    os.makedirs(POLICIES_DIR, exist_ok=True)
    
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
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
            # Convert position to one-hot encoding
            pos_onehot = position_to_onehot(pos)
            
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 3)  # explore
            else:
                with torch.no_grad():
                    qvals = net(torch.tensor([pos_onehot], dtype=torch.float32))
                    action = int(torch.argmax(qvals).item())

            # Map action index to environment delta
            # Action 0 = left, Action 1 = no move, Action 2 = right
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
            q_pred = net(torch.tensor([pos_onehot], dtype=torch.float32))[0, action]
            with torch.no_grad():
                next_pos_onehot = position_to_onehot(next_pos)
                q_next_max = net(torch.tensor([next_pos_onehot], dtype=torch.float32)).max()
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
            
            # Debug: check what the policy does at key positions
            debug_actions = []
            for debug_pos in [8, 10, 16]:
                debug_onehot = position_to_onehot(debug_pos)
                with torch.no_grad():
                    debug_qvals = net(torch.tensor([debug_onehot], dtype=torch.float32))
                    debug_action = int(torch.argmax(debug_qvals).item())
                debug_actions.append(f"pos{debug_pos}->act{debug_action}")
            
            await websocket.send_json({
                "type": "progress",
                "episode": ep + 1,
                "reward": round(total_reward, 3),
                "loss": round(avg_loss, 5),
                "debug": " | ".join(debug_actions),
            })

    # Export policy to ONNX
    dummy_input = torch.tensor([position_to_onehot(START_POS)], dtype=torch.float32)
    torch.onnx.export(
        net,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    await websocket.send_json({
        "type": "trained",
        "file_url": f"/policies/{model_filename}",
        "model_filename": model_filename,
        "timestamp": timestamp,
        "session_uuid": session_uuid,
    })

# -----------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------

def list_available_models():
    """List all available trained policy models."""
    if not os.path.exists(POLICIES_DIR):
        return []
    
    policy_files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("basic_policy_") and f.endswith(".onnx")]
    policy_files.sort(reverse=True)  # Most recent first
    return policy_files

# -----------------------------------------------------------------------------
# Inference helper -------------------------------------------------------------

_ort_sessions = {}  # cache for multiple models

async def infer_action(position: int, model_filename: str = None):
    """Run the trained ONNX policy and return an action index (0,1,2)."""
    global _ort_sessions
    
    # If no model specified, try to find the most recent one
    if model_filename is None:
        if not os.path.exists(POLICIES_DIR):
            raise FileNotFoundError(f"No policies directory found at {POLICIES_DIR}")
        
        # Find all policy files and get the most recent one
        policy_files = [f for f in os.listdir(POLICIES_DIR) if f.startswith("basic_policy_") and f.endswith(".onnx")]
        if not policy_files:
            raise FileNotFoundError("No trained policy files found")
        
        # Sort by timestamp (embedded in filename) and get the most recent
        policy_files.sort(reverse=True)
        model_filename = policy_files[0]
    
    model_path = os.path.join(POLICIES_DIR, model_filename)
    
    # Check if we already have this model loaded
    if model_filename not in _ort_sessions:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        import onnxruntime as ort
        _ort_sessions[model_filename] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Convert position to one-hot encoding like in training
    pos_onehot = position_to_onehot(position)
    inp = np.array([pos_onehot], dtype=np.float32)
    outputs = _ort_sessions[model_filename].run(None, {"input": inp})
    return int(np.argmax(outputs[0])) 