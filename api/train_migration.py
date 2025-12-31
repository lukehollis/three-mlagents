
import threading
import uvicorn
import time
import queue
import logging
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MigrationDemo")

# Updated Imports
from mlagents_bridge.env import ThreeJSEnv
from mlagents_bridge.server import create_server
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

def run_server(env):
    """Starts the uvicorn server in a separate thread."""
    env_map = {"default": env}
    app = create_server(env_map)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

def main():
    logger.info("Starting Migration Demo...")
    
    # 1. Create Side Channels
    env_params = EnvironmentParametersChannel()
    
    # 2. Create Environment
    # We pass the side channel list to the env
    env = ThreeJSEnv(channel_id="default", side_channels=[env_params])
    
    # 3. Start Server in Thread
    server_thread = threading.Thread(target=run_server, args=(env,), daemon=True)
    server_thread.start()
    
    logger.info("Server started on port 8000. Waiting for client to connect...")
    
    # 4. Wait for handshake and Reset
    try:
        env.reset()
        logger.info("Environment reset complete. Handshake successful.")
    except Exception as e:
        logger.error(f"Failed to reset: {e}")
        return

    # Set Initial Gravity
    env_params.set_float_parameter("gravity", 9.81)

    # 5. Run Loop
    behavior_name = "Ball3D"  # Must match JS
    
    try:
        for i in range(10000): # Run longer
            # Get decision steps
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Log Obs Shapes occasionally
            if i % 100 == 0:
                n_agents = len(decision_steps)
                logger.info(f"Step {i}: Agents={n_agents}")
                for idx, obs in enumerate(decision_steps.obs):
                    logger.info(f"  Obs[{idx}] Shape: {obs.shape}")

            # Create random actions
            if behavior_name in env.behavior_specs:
                spec = env.behavior_specs[behavior_name]
                n_agents = len(decision_steps)
                
                if n_agents > 0:
                    action = spec.action_spec.random_action(n_agents)
                    env.set_actions(behavior_name, action)
            
            # Step env
            env.step()
            
            # Test Side Channel
            if i % 100 == 0:
                new_g = 5.0 + np.random.rand() * 10.0
                logger.info(f"Step {i}: Setting Gravity to {new_g:.2f}")
                env_params.set_float_parameter("gravity", float(new_g))
                
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
