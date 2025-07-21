import asyncio
import random
import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from fastapi import WebSocket
import logging
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from services.llm import get_json, get_embedding

logger = logging.getLogger(__name__)

# Global websocket reference for logging to frontend
_current_websocket = None

RETRO_SCIFI_COLORS = [
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.6, 0.0],  # Bright Orange
    [0.7, 1.0, 0.0],  # Lime Green
    [0.1, 0.5, 1.0],  # Electric Blue
    [1.0, 1.0, 0.2],  # Bright Yellow
    [1.0, 0.2, 0.8],  # Pink
    [0.8, 0.0, 1.0],  # Purple
    [0.2, 1.0, 0.2],  # Green
]

def log_to_frontend(message: str):
    """Send log message to frontend InfoPanel if websocket is available"""
    if _current_websocket:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_current_websocket.send_json({
                "type": "debug", 
                "message": message
            }))

# --- Constants ---
NUM_PEDESTRIANS = 50
NUM_BUSINESSES = 20
MAX_MESSAGES = 20
MAX_LLM_LOGS = 30
LLM_CALL_FREQUENCY = 20
USE_LOCAL_OLLAMA = True 
MAX_STEPS_PER_EPISODE = 2000

# Economic Resources
RESOURCES = ["food", "money", "goods", "services", "tools", "materials"]
BUSINESS_TYPES = ["restaurant", "shop", "office", "factory", "market", "bank"]

# --- Entity Classes ---
class TrafficLight:
    def __init__(self, light_id: int, pos: np.ndarray, initial_state: str = 'green', cycle_time: int = 200):
        self.id = light_id
        self.pos = pos
        self.state = initial_state 
        self.cycle_time = cycle_time
        self.timer = random.randint(0, cycle_time)

    def step(self):
        self.timer += 1
        if self.timer >= self.cycle_time:
            self.timer = 0
            self.state = 'green' if self.state == 'red' else 'red'

class Business:
    def __init__(self, business_id: int, pos: np.ndarray, business_type: str):
        self.id = business_id
        self.pos = pos
        self.type = business_type
        self.inventory = {resource: random.randint(10, 50) for resource in RESOURCES}
        self.prices = {resource: random.uniform(1.0, 5.0) for resource in RESOURCES}
        self.customers_served = 0
        self.revenue = 0.0
        
    def serve_customer(self, customer_needs: Dict[str, int]) -> Dict[str, int]:
        """Serve a customer and return what was provided"""
        provided = {}
        cost = 0.0
        
        for resource, needed in customer_needs.items():
            if resource in self.inventory and self.inventory[resource] >= needed:
                self.inventory[resource] -= needed
                provided[resource] = needed
                cost += needed * self.prices[resource]
        
        if provided:
            self.customers_served += 1
            self.revenue += cost
            
        return provided, cost

class Pedestrian:
    def __init__(self, ped_id: int, start_pos: np.ndarray, graph: 'networkx.MultiDiGraph'):
        self.id = ped_id
        self.graph = graph
        self.pos = start_pos.copy()
        self.speed = random.uniform(0.5, 1.5)
        self.color = random.choice(RETRO_SCIFI_COLORS)
        
        # Economic attributes
        self.resources = {resource: random.randint(0, 20) for resource in RESOURCES}
        self.resources["money"] = random.randint(50, 200)
        self.needs = {resource: random.randint(1, 5) for resource in RESOURCES[:4]}  # Basic needs
        self.satisfaction = 50.0
        self.job_type = random.choice(BUSINESS_TYPES)
        self.memory_stream = []
        
        # Movement attributes
        self.target_pos = None
        self.target_business = None
        self.state = random.choice(['wandering', 'shopping', 'working', 'traveling'])
        self.state_timer = 0
        self.path_progress = 0.0
        
        # Find nearest nodes for pathfinding
        self._find_nearest_nodes()
        
    def _find_nearest_nodes(self):
        """Find the nearest graph nodes to current position"""
        min_dist = float('inf')
        self.current_node = None
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_pos = np.array([node_data['y'], node_data['x']])
            dist = np.linalg.norm(self.pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                self.current_node = node_id
    
    def set_destination(self, target_pos: np.ndarray):
        """Set a destination and find path"""
        # Find nearest node to target
        min_dist = float('inf')
        target_node = None
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_pos = np.array([node_data['y'], node_data['x']])
            dist = np.linalg.norm(target_pos - node_pos)
            if dist < min_dist:
                min_dist = dist
                target_node = node_id
        
        if target_node and self.current_node:
            try:
                path = ox.shortest_path(self.graph, self.current_node, target_node, weight='length')
                if path and len(path) > 1:
                    self.path = path
                    self.path_index = 0
                    self.target_pos = target_pos
                    self.path_progress = 0.0
                    return True
            except Exception:
                pass
        
        return False
    
    def step(self, businesses: List[Business], traffic_lights: List[TrafficLight]):
        """Update pedestrian state and position"""
        self.state_timer += 1
        
        # State transitions
        if self.state == 'wandering' and self.state_timer > 50:
            if random.random() < 0.3:
                self.state = 'shopping'
                self.state_timer = 0
        elif self.state == 'shopping' and self.state_timer > 100:
            if random.random() < 0.4:
                self.state = 'working'
                self.state_timer = 0
        elif self.state == 'working' and self.state_timer > 200:
            if random.random() < 0.5:
                self.state = 'wandering'
                self.state_timer = 0
        
        # Behavior based on state
        if self.state == 'shopping' and not self.target_pos:
            # Find nearest business
            nearest_business = min(businesses, key=lambda b: np.linalg.norm(self.pos - b.pos))
            if np.linalg.norm(self.pos - nearest_business.pos) > 0.0001:
                self.set_destination(nearest_business.pos)
                self.target_business = nearest_business
        
        elif self.state == 'working' and not self.target_pos:
            # Find a business of their job type
            job_businesses = [b for b in businesses if b.type == self.job_type]
            if job_businesses:
                target = random.choice(job_businesses)
                self.set_destination(target.pos)
        
        elif self.state == 'wandering' and not self.target_pos:
            # Pick a random destination
            all_nodes = list(self.graph.nodes())
            if all_nodes:
                random_node = random.choice(all_nodes)
                node_data = self.graph.nodes[random_node]
                random_pos = np.array([node_data['y'], node_data['x']])
                self.set_destination(random_pos)
        
        # Movement
        if hasattr(self, 'path') and hasattr(self, 'path_index'):
            self._move_along_path()
            
        # Economic interactions
        if self.target_business and np.linalg.norm(self.pos - self.target_business.pos) < 0.0001:
            self._interact_with_business(self.target_business)
            self.target_business = None
            self.target_pos = None
            
        # Update satisfaction based on needs
        self._update_satisfaction()
    
    def _move_along_path(self):
        """Move along the current path"""
        if not hasattr(self, 'path') or self.path_index >= len(self.path) - 1:
            return
            
        p1 = self.graph.nodes[self.path[self.path_index]]
        p2 = self.graph.nodes[self.path[self.path_index + 1]]
        
        start_pos = np.array([p1['y'], p1['x']])
        end_pos = np.array([p2['y'], p2['x']])
        
        # Move towards next node
        direction = end_pos - start_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction_norm = direction / distance
            movement = direction_norm * self.speed * 0.00001  # Scale for lat/lng
            self.pos += movement
            
            # Check if reached next node
            if np.linalg.norm(self.pos - end_pos) < 0.00001:
                self.path_index += 1
                self.current_node = self.path[self.path_index]
                
                if self.path_index >= len(self.path) - 1:
                    # Reached destination
                    self.target_pos = None
    
    def _interact_with_business(self, business: Business):
        """Interact with a business"""
        if self.state == 'shopping':
            # Try to buy things we need
            needed = {resource: need for resource, need in self.needs.items() 
                     if self.resources[resource] < need}
            
            if needed and self.resources["money"] > 0:
                provided, cost = business.serve_customer(needed)
                
                if provided and self.resources["money"] >= cost:
                    self.resources["money"] -= cost
                    for resource, amount in provided.items():
                        self.resources[resource] += amount
                        if resource in self.needs:
                            self.needs[resource] = max(0, self.needs[resource] - amount)
                    
                    self.add_to_memory_stream(f"Bought {provided} from {business.type} for ${cost:.1f}")
        
        elif self.state == 'working':
            # Work and earn money
            if business.type == self.job_type:
                payment = random.uniform(10, 30)
                self.resources["money"] += payment
                business.revenue -= payment  # Business pays worker
                self.add_to_memory_stream(f"Worked at {business.type} and earned ${payment:.1f}")
    
    def _update_satisfaction(self):
        """Update satisfaction based on needs fulfillment"""
        total_needs = sum(self.needs.values())
        if total_needs > 0:
            self.satisfaction = max(0, min(100, 100 - total_needs * 5))
        else:
            self.satisfaction = min(100, self.satisfaction + 1)
    
    def add_to_memory_stream(self, event: str):
        """Add event to memory stream"""
        self.memory_stream.append(event)
        if len(self.memory_stream) > 10:
            self.memory_stream.pop(0)

# --- Environment Class ---
class SimCityEnv:
    def __init__(self):
        self.llm_logs: List[Dict] = []
        self.messages: List[Dict] = []
        self.step_count = 0
        self.pedestrians: List[Pedestrian] = []
        self.businesses: List[Business] = []
        self.traffic_lights: List[TrafficLight] = []
        self.running = False
        
        # Load real city data
        self.location_point = (37.7749, -122.4194)  # Downtown San Francisco
        self.graph = ox.graph_from_point(self.location_point, dist=900, network_type='drive')
        
        # Add elevation data if possible
        try:
            google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
            if google_api_key:
                ox.add_node_elevations_google(self.graph, api_key=google_api_key)
        except Exception as e:
            logger.warning(f"Could not get elevation data: {e}")
        
        self.road_network_for_viz = self._get_road_network_for_viz()
        self._create_businesses()
        self._create_traffic_lights()
        self._create_pedestrians()

    def _get_road_network_for_viz(self):
        """Convert road network to visualization format"""
        lines = []
        for u, v, data in self.graph.edges(data=True):
            if 'geometry' in data:
                xs, ys = data['geometry'].xy
                lines.append([[ys[i], xs[i]] for i in range(len(xs))])
        return lines

    def _create_businesses(self):
        """Create businesses at strategic locations"""
        # Find high-degree nodes (intersections) for business placement
        high_degree_nodes = [node for node, degree in self.graph.degree() if degree >= 3]
        
        # Select random subset for businesses
        business_nodes = random.sample(high_degree_nodes, min(NUM_BUSINESSES, len(high_degree_nodes)))
        
        for i, node_id in enumerate(business_nodes):
            node_data = self.graph.nodes[node_id]
            business_pos = np.array([node_data['y'], node_data['x']])
            business_type = random.choice(BUSINESS_TYPES)
            
            # Add small offset to avoid overlap
            offset = np.array([random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)])
            business_pos += offset
            
            self.businesses.append(Business(i, business_pos, business_type))

    def _create_traffic_lights(self):
        """Create traffic lights at major intersections"""
        intersections = [node for node, degree in self.graph.degree() if degree > 3]
        
        # Limit to avoid performance issues
        selected_intersections = random.sample(intersections, min(10, len(intersections)))
        
        for i, node_id in enumerate(selected_intersections):
            node_data = self.graph.nodes[node_id]
            light_pos = np.array([node_data['y'], node_data['x']])
            self.traffic_lights.append(TrafficLight(i, light_pos))

    def _create_pedestrians(self):
        """Create pedestrians at random locations"""
        all_nodes = list(self.graph.nodes())
        
        for i in range(NUM_PEDESTRIANS):
            # Pick random starting position
            start_node = random.choice(all_nodes)
            node_data = self.graph.nodes[start_node]
            start_pos = np.array([node_data['y'], node_data['x']])
            
            # Add small random offset
            offset = np.array([random.uniform(-0.0001, 0.0001), random.uniform(-0.0001, 0.0001)])
            start_pos += offset
            
            pedestrian = Pedestrian(i, start_pos, self.graph)
            self.pedestrians.append(pedestrian)

    def add_message(self, ped_id: int, message: str):
        """Add a message to the environment's message list"""
        if len(self.messages) > MAX_MESSAGES:
            self.messages.pop(0)
        self.messages.append({
            "sender_id": ped_id,
            "recipient_id": None,
            "message": message,
            "step": self.step_count
        })

    def step(self):
        """Execute one simulation step"""
        self.step_count += 1
        
        # Update traffic lights
        for light in self.traffic_lights:
            light.step()
        
        # Update pedestrians
        for pedestrian in self.pedestrians:
            pedestrian.step(self.businesses, self.traffic_lights)
            
            # Occasionally add messages about their activities
            if random.random() < 0.05:  # 5% chance per step
                if pedestrian.memory_stream:
                    latest_activity = pedestrian.memory_stream[-1]
                    self.add_message(pedestrian.id, f"Satisfaction: {pedestrian.satisfaction:.0f}% - {latest_activity}")

    def get_state_for_viz(self) -> Dict[str, Any]:
        """Get current state for visualization"""
        return {
            "pedestrians": [{
                "id": p.id,
                "pos": p.pos.tolist(),
                "color": p.color,
                "state": p.state,
                "satisfaction": p.satisfaction,
                "resources": p.resources,
                "memory_stream": p.memory_stream
            } for p in self.pedestrians],
            "businesses": [{
                "id": b.id,
                "pos": b.pos.tolist(),
                "type": b.type,
                "inventory": b.inventory,
                "customers_served": b.customers_served,
                "revenue": b.revenue
            } for b in self.businesses],
            "traffic_lights": [{
                "id": l.id,
                "pos": l.pos.tolist(),
                "state": l.state
            } for l in self.traffic_lights],
            "road_network": self.road_network_for_viz,
            "messages": self.messages,
            "llm_logs": self.llm_logs
        }

# --- Websocket handlers ---
async def run_simcity(websocket: WebSocket, env: SimCityEnv):
    """Run the simulation"""
    global _current_websocket
    _current_websocket = websocket
    
    env.running = True
    step_rewards = []

    try:
        while env.running:
            env.step()
            
            # Send state update
            state = env.get_state_for_viz()
            await websocket.send_json({"type": "run_step", "state": state})
            
            # Calculate some metrics for charts
            avg_satisfaction = np.mean([p.satisfaction for p in env.pedestrians])
            total_business_revenue = sum([b.revenue for b in env.businesses])
            
            step_rewards.append(avg_satisfaction)
            
            await websocket.send_json({
                "type": "progress",
                "episode": env.step_count,
                "reward": avg_satisfaction,
                "loss": None
            })
            
            await asyncio.sleep(0.1)  # Control simulation speed
            
    except asyncio.CancelledError:
        logger.info("SimCity simulation cancelled")
    except Exception as e:
        logger.error(f"Error in SimCity simulation: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Simulation failed: {e}"})
    finally:
        env.running = False

async def train_simcity(websocket: WebSocket, env: SimCityEnv):
    """Train the simulation (placeholder for future ML training)"""
    global _current_websocket
    _current_websocket = websocket
    
    try:
        # For now, just run a longer simulation as "training"
        await websocket.send_json({"type": "training_progress", "message": "Starting economic simulation training..."})
        
        for epoch in range(100):  # Simulate training epochs
            env.step()
            
            if epoch % 10 == 0:
                state = env.get_state_for_viz()
                await websocket.send_json({"type": "train_step", "state": state, "epoch": epoch})
                
                avg_satisfaction = np.mean([p.satisfaction for p in env.pedestrians])
                await websocket.send_json({
                    "type": "progress",
                    "episode": epoch,
                    "reward": avg_satisfaction,
                    "loss": 100 - avg_satisfaction  # Mock loss
                })
            
            await asyncio.sleep(0.05)
        
        await websocket.send_json({
            "type": "trained",
            "model_info": {
                "epochs": 100,
                "final_satisfaction": np.mean([p.satisfaction for p in env.pedestrians])
            }
        })
        
    except Exception as e:
        logger.error(f"Error during SimCity training: {e}", exc_info=True)
        await websocket.send_json({"type": "error", "message": f"Training failed: {e}"})


