
import logging
import queue
import time
import uuid
import base64
import numpy as np
import io
try:
    from PIL import Image
except ImportError:
    Image = None

from typing import Dict, Any, List, Tuple, Optional

from mlagents_envs.base_env import (
    BaseEnv,
    BehaviorSpec,
    ActionTuple,
    DecisionSteps,
    TerminalSteps,
    ActionSpec,
    ObservationSpec,
    ObservationType,
    DimensionProperty,
)
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.side_channel_manager import SideChannelManager

logger = logging.getLogger("ThreeJSEnv")

class ThreeJSEnv(BaseEnv):
    def __init__(self, channel_id: str = "default", side_channels: Optional[List[SideChannel]] = None):
        super().__init__()
        self.channel_id = channel_id
        
        # Side Channels
        if side_channels is None:
            side_channels = []
        self._side_channel_manager = SideChannelManager(side_channels)

        # Communication queues (shared with the WebSocket handler)
        # Inbound: Messages FROM browser TO python
        self.inbound_queue = queue.Queue()
        # Outbound: Messages FROM python TO browser
        self.outbound_queue = queue.Queue()
        
        # State
        self.behavior_specs: Dict[str, BehaviorSpec] = {}
        self.latest_step_data: Dict[str, Any] = {} # Map agent_id -> data
        self.agents_action_input: Dict[str, ActionTuple] = {}
        
        self.waiting_for_handshake = True
        self.connected = False
        
        logger.info(f"ThreeJSEnv created. Waiting for WebSocket connection on channel {channel_id}...")

    def set_queues(self, inbound: queue.Queue, outbound: queue.Queue):
        """Called by the Server/Main to link the websocket queues."""
        self.inbound_queue = inbound
        self.outbound_queue = outbound
        self.connected = True
        
    def _poll_inbound(self, timeout: float = None) -> Optional[Dict]:
        try:
            return self.inbound_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def step(self) -> None:
        """
        Signals the environment that it must move the simulation forward by one step.
        """
        # 1. Send actions from previous set_actions calls to the browser
        actions_payload = {}
        for behavior_name, action_tuple in self.agents_action_input.items():
            # For now, we mix all agents from all behaviors.
            # We need to map the batched action tuple back to individual Agent IDs.
            # This requires us to know which Agent ID corresponded to which index in the previous DecisionStep.
            # This state tracking is complex in BaseEnv.
            # Simplified approach: We assume we stored the agent_ids associated with the batches during get_steps.
            pass

        # Actually, simpler: ThreeJSEnv stores actions in a dict keyed by AgentID.
        # But set_actions provides batch data.
        # We need to rely on the fact that the user calls set_actions matching the agent_order of the last get_steps.
        
        # FOR PROTOTYPE: We will construct a simple payload.
        # But wait, BaseEnv API is strict.
        # Let's send the "step" command to browser, including any actions queued up.
        
        # Construct action payload
        # simple map: agent_id -> action
        # We need to resolve the batch back to IDs.
        
        # Let's perform the "Send" logic.
        action_map = self._resolve_actions_to_map()
        
        # Generate Side Channel Data (Bytes -> Base64)
        side_channel_data = self._side_channel_manager.generate_side_channel_messages()
        side_channel_b64 = base64.b64encode(side_channel_data).decode('utf-8') if side_channel_data else None
        
        msg = {
            "type": "step",
            "actions": action_map,
            "side_channels": side_channel_b64
        }
        self.outbound_queue.put(msg)
        
        # 2. Wait for the browser to reply with new observations
        # This blocks until browser responds.
        response = self._poll_inbound(timeout=30.0) # 30s timeout
        if not response:
            raise TimeoutError("Timed out waiting for browser step response.")
        
        if response["type"] == "step":
            self.latest_step_data = response["agents"]
            
            # Process inbound side channels
            if "side_channels" in response and response['side_channels']:
                try:
                    sc_bytes = base64.b64decode(response['side_channels'])
                    self._side_channel_manager.process_side_channel_message(sc_bytes)
                except Exception as e:
                    logger.error(f"Failed to process inbound side channels: {e}")
        else:
            logger.warning(f"Unexpected message during step: {response}")

        # Clear actions buffer for next cycle
        self.agents_action_input = {}
        self._batch_agent_ids = {} # clear cache

    def reset(self) -> None:
        """
        Signals the environment that it must reset the simulation.
        """
        # Check if we have handshaked
        if self.waiting_for_handshake:
            logger.info("Waiting for handshake...")
            # We expect the first message to be handshake
            while True:
                msg = self._poll_inbound(timeout=60)
                if msg and msg["type"] == "handshake":
                    self._process_handshake(msg)
                    break
        
        # Generate Side Channel Data (Bytes -> Base64)
        side_channel_data = self._side_channel_manager.generate_side_channel_messages()
        side_channel_b64 = base64.b64encode(side_channel_data).decode('utf-8') if side_channel_data else None

        self.outbound_queue.put({
            "type": "reset",
            "side_channels": side_channel_b64
        })
        
        # Perform one step to get initial observations
        # Wait for "step" response (initial obs)
        response = self._poll_inbound(timeout=10.0)
        start_time = time.time()
        while not response or response.get("type") != "step":
            if time.time() - start_time > 10:
                raise TimeoutError("Failed to get initial observations after reset")
            response = self._poll_inbound(timeout=1)

        self.latest_step_data = response["agents"]
        
        # Process inbound side channels (from reset response)
        if "side_channels" in response and response['side_channels']:
            try:
                sc_bytes = base64.b64decode(response['side_channels'])
                self._side_channel_manager.process_side_channel_message(sc_bytes)
            except Exception as e:
                logger.error(f"Failed to process inbound side channels: {e}")

    def close(self) -> None:
        self.connected = False

    @property
    def behavior_specs(self) -> Dict[str, BehaviorSpec]:
        return self._behavior_specs

    def set_actions(self, behavior_name: str, action: ActionTuple) -> None:
        if behavior_name not in self.agents_action_input:
            self.agents_action_input[behavior_name] = action
        else:
            # Append or overwrite? Usually set_actions is called once per behavior per step.
            self.agents_action_input[behavior_name] = action

    def set_action_for_agent(self, behavior_name: str, agent_id: int, action: ActionTuple) -> None:
        # Not implemented for batched training usually, but we can store it.
        pass

    def get_steps(self, behavior_name: str) -> Tuple[DecisionSteps, TerminalSteps]:
        # Filter latest_step_data for agents with this behavior
        agents_data = []
        for agent_id, data in self.latest_step_data.items():
            # In handshake/step, we should track which agent has which behavior.
            # For prototype, assume all agents have the same behavior or we check metadata.
            # Let's assume we map all 'default' agents.
            # We need to know the behavior of each agent_id.
            if self._agent_behavior_map.get(agent_id) == behavior_name:
                agents_data.append((agent_id, data))
        
        # Separate into Decision (running) and Terminal (done)
        decision_agents = []
        terminal_agents = []
        
        for aid, data in agents_data:
            if data['done']:
                terminal_agents.append((aid, data))
            else:
                decision_agents.append((aid, data))
                
        # Construct DecisionSteps
        d_obs_list = []
        d_rewards = []
        d_ids = []
        
        if decision_agents:
            n_agents = len(decision_agents)
            
            # 1. Vector Obs
            # Assume all agents have 'vectorObs'
            vec_batch = []
            for _, d in decision_agents:
                # Fallback to old 'obs' if key missing
                vec_batch.append(d.get('vectorObs', d.get('obs', [])))
            
            d_obs_list.append(np.array(vec_batch, dtype=np.float32))

            # 2. Visual Obs (from 'visualObs' list of b64 strings)
            # We assume all agents have same number of visual obs (e.g. 1)
            # We need to know the shape from spec to initialize array?
            # Or we decode first.
            
            # Check first agent to see how many visuals
            first_data = decision_agents[0][1]
            n_visuals = len(first_data.get('visualObs', []))
            
            for vis_idx in range(n_visuals):
                vis_batch = []
                for _, d in decision_agents:
                    b64_str = d['visualObs'][vis_idx]
                    if b64_str and Image:
                        try:
                            img = Image.open(io.BytesIO(base64.b64decode(b64_str)))
                            img = img.convert("RGB")
                            # Resize if needed? We assume client sends correct size.
                            # Standard ML-Agents expects (H, W, 3) 0-1 float
                            arr = np.array(img, dtype=np.float32) / 255.0
                            vis_batch.append(arr)
                        except Exception as e:
                            logger.error(f"Image decode error: {e}")
                            vis_batch.append(np.zeros((1,1,3), dtype=np.float32)) # error placeholder
                    else:
                        vis_batch.append(np.zeros((1,1,3), dtype=np.float32)) # empty/no PIL

                d_obs_list.append(np.array(vis_batch, dtype=np.float32))

            d_rewards = np.array([d['reward'] for _, d in decision_agents], dtype=np.float32)
            d_ids = np.array([self._agent_id_to_int(aid) for aid, _ in decision_agents], dtype=np.int32)
            
            # Cache ids to solve set_actions mapping later
            self._batch_agent_ids[behavior_name] = d_ids
            
            ds = DecisionSteps(d_obs_list, d_rewards, d_ids, None, np.zeros(n_agents, dtype=np.int32), np.zeros(n_agents, dtype=np.float32))
        else:
             ds = DecisionSteps.empty(self.behavior_specs[behavior_name])

        # Construct TerminalSteps
        if terminal_agents:
            t_obs_list = [np.array([d['obs'] for _, d in terminal_agents], dtype=np.float32)]
            t_rewards = np.array([d['reward'] for _, d in terminal_agents], dtype=np.float32)
            t_interrupted = np.zeros(len(terminal_agents), dtype=bool) # Basic doesn't support interrupted yet
            t_ids = np.array([self._agent_id_to_int(aid) for aid, _ in terminal_agents], dtype=np.int32)
            
            ts = TerminalSteps(t_obs_list, t_rewards, t_interrupted, t_ids, np.zeros(len(terminal_agents), dtype=np.int32), np.zeros(len(terminal_agents), dtype=np.float32))
        else:
             ts = TerminalSteps.empty(self.behavior_specs[behavior_name])
             
        return ds, ts

    # --- Helpers ---

    def _process_handshake(self, msg):
        self._behavior_specs = {}
        self._agent_behavior_map = {} # agent_id (str) -> behavior_name
        self.waiting_for_handshake = False
        
        # msg.behaviors: { "BehaviorName": { observationSpecs: ..., actionSpec: ... } }
        for name, stats in msg["behaviors"].items():
            # Parse Observation Specs
            obs_specs = []
            for ospec in stats["observationSpecs"]:
                # ospec: { shape: [8], name: "VectorSensor" }
                obs_specs.append(ObservationSpec(
                    shape=tuple(ospec["shape"]),
                    dimension_property=(DimensionProperty.NONE,), # simplify
                    observation_type=ObservationType.DEFAULT,
                    name=ospec.get("name", "Observation")
                ))
            
            # Parse Action Spec
            aspec_data = stats["actionSpec"]
            if aspec_data.get("discrete") and len(aspec_data["discrete"]) > 0:
                 action_spec = ActionSpec.create_discrete(tuple(aspec_data["discrete"]))
            elif aspec_data["continuous"] > 0:
                 action_spec = ActionSpec.create_continuous(aspec_data["continuous"])
            else:
                 action_spec = ActionSpec.create_continuous(0) #?

            self._behavior_specs[name] = BehaviorSpec(obs_specs, action_spec)
            
            # Note: We don't know agents yet, they appear in 'step'
            
        logger.info(f"Handshake complete. Behaviors: {list(self._behavior_specs.keys())}")

    def _resolve_actions_to_map(self):
        # Convert batched actions in agents_action_input back to { agent_id: action }
        action_map = {}
        
        for name, action_tuple in self.agents_action_input.items():
            agent_ids = self._batch_agent_ids.get(name, [])
            if len(agent_ids) == 0:
                continue
            
            # Is it continuous or discrete?
            # action_tuple.continuous (N, C)
            # action_tuple.discrete (N, D)
            
            for i, aid_int in enumerate(agent_ids):
                aid_str = self._int_to_agent_id_map.get(aid_int)
                if not aid_str: continue
                
                # Extract action for this agent
                # Send generic structure, let JS handle it
                # For simplicity, if continuous > 0 send that, else discrete
                act_data = {}
                if action_tuple.continuous.shape[1] > 0:
                    act_data = action_tuple.continuous[i].tolist()
                elif action_tuple.discrete.shape[1] > 0:
                     # If only 1 branch, send single int, else list
                     d = action_tuple.discrete[i].tolist()
                     act_data = d[0] if len(d) == 1 else d
                
                action_map[aid_str] = act_data
        
        return action_map

    # Setup ID mapping because ML-Agents uses Int IDs but we use UUIDs
    _int_counter = 1
    _agent_id_to_int_map = {}
    _int_to_agent_id_map = {}
    
    def _agent_id_to_int(self, aid_str: str) -> int:
        if aid_str not in self._agent_id_to_int_map:
            self._agent_id_to_int_map[aid_str] = self._int_counter
            self._int_to_agent_id_map[self._int_counter] = aid_str
            self._int_counter += 1
            
            # Also need to track its behavior. 
            # For now assume we look it up or defaults.
            # In handshake/step protocol we might need agent to send its behavior name every time 
            # or we assume 'Default' for everything if unspecified.
            self._agent_behavior_map[aid_str] = "Default" # Hack for now
            
        return self._agent_id_to_int_map[aid_str]

