import random
import numpy as np

# --- Environment Constants ---
GRID_SIZE = 64
NUM_VEHICLES = 16
REWARD_PROGRESS = 1.0
REWARD_COLLISION = -50.0
REWARD_STEP = -0.1
VEHICLE_MIN_SPEED = 0.5
VEHICLE_MAX_SPEED = 2.0
ACCELERATION = 0.2
REWARD_RED_LIGHT = -25.0


# --- Traffic Light Definitions ---
class TrafficLightController:
    def __init__(self, cycle_time=200):
        # 0: NS Green, EW Red | 1: NS Red, EW Green
        self.state = 0
        self.cycle_time = cycle_time
        self.timer = 0

    def update(self):
        self.timer += 1
        if self.timer >= self.cycle_time:
            self.state = 1 - self.state
            self.timer = 0


INTERSECTIONS = {
    "main": {"pos": np.array([0, 0, 0]), "radius": 10.0, "controlled_by": "NS"},
    "secondary": {"pos": np.array([-25, 0, 0]), "radius": 8.0, "controlled_by": "EW"},
    "third": {"pos": np.array([20, 0, 0]), "radius": 8.0, "controlled_by": "EW"},
}

# --- Waypoint and Path Definitions ---
WAYPOINTS = {
    # Primary Roads
    "H_E": np.array([40.0, 0.0, 0.0]),
    "H_W": np.array([-40.0, 0.0, 0.0]),
    "V_N": np.array([0.0, 0.0, 20.0]),
    "I_CENTER": np.array([0.0, 0.0, 0.0]),
    # Feeder roads
    "TR_N": np.array([20.0, 0.0, 20.0]),
    "TR_I": np.array([20.0, 0.0, 0.0]),
    "BL_S": np.array([-25.0, 0.0, -20.0]),
    "BL_I": np.array([-25.0, 0.0, 0.0]),
    # Curved road from South-East
    "CR_S": np.array([25.0, 0.0, -20.0]),
    "CR_M": np.array([10.0, 0.0, -10.0]),
}


PATHS = {
    # E-W Traffic
    "E_to_W": (["H_E", "TR_I", "I_CENTER", "BL_I", "H_W"], "EW"),
    "W_to_E": (["H_W", "BL_I", "I_CENTER", "TR_I", "H_E"], "EW"),
    # From South (Curve)
    "S_Curve_to_N": (["CR_S", "CR_M", "I_CENTER", "V_N"], "NS"),
    "S_Curve_to_W": (["CR_S", "CR_M", "I_CENTER", "BL_I", "H_W"], "NS"),
    "S_Curve_to_E": (["CR_S", "CR_M", "I_CENTER", "TR_I", "H_E"], "NS"),
    # From North
    "N_to_S_Curve": (["V_N", "I_CENTER", "CR_M", "CR_S"], "NS"),
    "N_to_W": (["V_N", "I_CENTER", "BL_I", "H_W"], "NS"),
    "N_to_E": (["V_N", "I_CENTER", "TR_I", "H_E"], "NS"),
    # From feeder roads
    "TR_N_to_W": (["TR_N", "TR_I", "I_CENTER", "BL_I", "H_W"], "EW"),
    "TR_N_to_S": (["TR_N", "TR_I", "I_CENTER", "CR_M", "CR_S"], "EW"),
    "BL_S_to_E": (["BL_S", "BL_I", "I_CENTER", "TR_I", "H_E"], "EW"),
    "BL_S_to_N": (["BL_S", "BL_I", "I_CENTER", "V_N"], "EW"),
}


# --- Action Definitions ---
# 0: decelerate, 1: maintain speed, 2: accelerate
DISCRETE_ACTIONS = ["decelerate", "maintain", "accelerate"]


def _nearest_other_distance(positions: np.ndarray, index: int, fallback: float = 30.0):
    if len(positions) <= 1:
        return fallback
    distances = np.linalg.norm(positions - positions[index], axis=1)
    distances[index] = np.inf
    nearest = float(np.min(distances))
    return nearest if np.isfinite(nearest) else fallback


def _collision_pairs(positions: np.ndarray, radius: float):
    if len(positions) <= 1:
        return []
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    rows, cols = np.where(np.triu(distances < radius, k=1))
    return zip(rows.tolist(), cols.tolist())


# --- Environment Class ---
class MultiVehicleEnv:
    def __init__(self):
        self.vehicles = []
        self.step_count = 0
        self.light_controller = TrafficLightController()
        self.reset()

    def reset(self):
        self.step_count = 0
        self.vehicles = []
        for i in range(NUM_VEHICLES):
            self.spawn_vehicle(i)
        return self._get_states()

    def spawn_vehicle(self, vehicle_id):
        path_name = random.choice(list(PATHS.keys()))
        path_wps, path_group = PATHS[path_name]

        # Remove existing vehicle with same ID
        self.vehicles = [v for v in self.vehicles if v["id"] != vehicle_id]

        self.vehicles.append(
            {
                "id": vehicle_id,
                "pos": np.copy(WAYPOINTS[path_wps[0]]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "speed": VEHICLE_MIN_SPEED,
                "path_name": path_name,
                "path_wps": path_wps,
                "path_group": path_group,
                "wp_idx": 1,
            }
        )

    def _get_states(self):
        if not self.vehicles:
            return np.zeros((0, 7))  # Return empty if no vehicles

        states = np.zeros((len(self.vehicles), 7), dtype=np.float32)
        vehicle_positions = np.array([v["pos"] for v in self.vehicles])

        for i, v in enumerate(self.vehicles):
            # Vector to next waypoint
            target_wp = WAYPOINTS[v["path_wps"][v["wp_idx"]]]
            vec_to_wp = target_wp - v["pos"]

            # Distance to nearest vehicle
            nearest_dist = _nearest_other_distance(vehicle_positions, i)

            # Find next intersection and light state
            dist_to_light = 100.0
            light_state = 0.0  # 0: no light, 1: green, -1: red
            for name, intersection in INTERSECTIONS.items():
                dist_to_isect = np.linalg.norm(v["pos"] - intersection["pos"])
                if dist_to_isect < 40.0:  # Only consider nearby lights
                    if dist_to_isect < dist_to_light:
                        dist_to_light = dist_to_isect
                        # Is our path group allowed to go?
                        # state 0 = NS green, state 1 = EW green
                        is_ns_path = v["path_group"] == "NS"
                        is_ns_green = self.light_controller.state == 0

                        if (is_ns_path and is_ns_green) or (
                            not is_ns_path and not is_ns_green
                        ):
                            light_state = 1.0  # Green
                        else:
                            light_state = -1.0  # Red

            states[i, 0] = v["speed"]
            states[i, 1:4] = (
                vec_to_wp / np.linalg.norm(vec_to_wp)
                if np.linalg.norm(vec_to_wp) > 0
                else vec_to_wp
            )
            states[i, 4] = nearest_dist
            states[i, 5] = light_state
            states[i, 6] = dist_to_light / 40.0  # Normalized

        return states

    def step(self, actions):
        rewards = np.full(len(self.vehicles), REWARD_STEP)
        self.light_controller.update()

        for i, v in enumerate(self.vehicles):
            # Check for red light violation
            for name, intersection in INTERSECTIONS.items():
                dist_to_isect = np.linalg.norm(v["pos"] - intersection["pos"])
                if dist_to_isect < intersection["radius"]:
                    is_ns_path = v["path_group"] == "NS"
                    is_ew_path = v["path_group"] == "EW"
                    is_ns_green = self.light_controller.state == 0
                    is_ew_green = self.light_controller.state == 1

                    if (is_ns_path and not is_ns_green) or (
                        is_ew_path and not is_ew_green
                    ):
                        rewards[i] += REWARD_RED_LIGHT

            # 1. Update speed based on action
            action = actions[i]
            if action == 0:  # decelerate
                v["speed"] = max(VEHICLE_MIN_SPEED, v["speed"] - ACCELERATION)
            elif action == 2:  # accelerate
                v["speed"] = min(VEHICLE_MAX_SPEED, v["speed"] + ACCELERATION)

            # 2. Move vehicle towards next waypoint
            target_wp_pos = WAYPOINTS[v["path_wps"][v["wp_idx"]]]
            direction = target_wp_pos - v["pos"]
            dist_to_wp = np.linalg.norm(direction)

            if dist_to_wp > 0:
                v["velocity"] = (direction / dist_to_wp) * v["speed"]

            v["pos"] += v["velocity"]

            # 3. Check if waypoint is reached
            if np.linalg.norm(target_wp_pos - v["pos"]) < v["speed"]:  # If close enough
                if v["wp_idx"] < len(v["path_wps"]) - 1:
                    v["wp_idx"] += 1
                else:  # End of path
                    rewards[i] += REWARD_PROGRESS * 20  # Bonus for finishing
                    self.spawn_vehicle(v["id"])
                    continue  # skip collision check for respawned car

        # 4. Collision detection
        if len(self.vehicles) > 1:
            vehicle_positions_after = np.array([v["pos"] for v in self.vehicles])
            collisions = _collision_pairs(vehicle_positions_after, radius=1.5)

            collided_indices = set()
            for c1, c2 in collisions:
                rewards[c1] += REWARD_COLLISION
                rewards[c2] += REWARD_COLLISION
                collided_indices.add(c1)
                collided_indices.add(c2)

            for idx in collided_indices:
                self.spawn_vehicle(self.vehicles[idx]["id"])

        self.step_count += 1
        done = self.step_count >= 1000  # Episode ends after N steps

        return self._get_states(), rewards, done

    def get_state_for_viz(self):
        agents = []
        for v in self.vehicles:
            agents.append(
                {
                    "id": v["id"],
                    "pos": [float(v["pos"][0]), float(v["pos"][1]), float(v["pos"][2])],
                    "energy": (v["speed"] / VEHICLE_MAX_SPEED) * 100,
                    "velocity": [
                        float(v["velocity"][0]),
                        float(v["velocity"][1]),
                        float(v["velocity"][2]),
                    ],
                }
            )

        return {"agents": agents, "lights": self.light_controller.state}
