from fastapi import WebSocket

# Environment constants
SMALL_GOAL = 7
LARGE_GOAL = 17
MIN_POS = 0
MAX_POS = 20
START_POS = 10

# Base path for policies directory
POLICIES_DIR = "policies"

# ----------------------------------------------------------------------------
# Standardized SB3 WebSocket entry points
# ----------------------------------------------------------------------------


async def train_basic(websocket: WebSocket):
    from mlagents.websocket_training import train_task_for_websocket

    await train_task_for_websocket(
        websocket,
        "basic",
        total_timesteps=25_000,
        algorithm="dqn",
        progress_freq=500,
    )


async def infer_action(position: int, model_filename: str | None = None):
    from mlagents.envs import position_to_onehot
    from mlagents.websocket_training import predict_discrete_action

    return predict_discrete_action(
        "basic", position_to_onehot(position), model_filename
    )
