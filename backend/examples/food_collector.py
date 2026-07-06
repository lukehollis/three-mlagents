from typing import List, Dict, Any, Tuple

import numpy as np

# -----------------------------------------------------------------------------------
# Food Collector Environment
# -----------------------------------------------------------------------------------

NUM_AGENTS = 5
AREA_SIZE = 40
NUM_GOOD_FOOD = 10
NUM_BAD_FOOD = 3
AGENT_RADIUS = 1.0
FOOD_RADIUS = 0.5
LASER_LENGTH = 25.0
FROZEN_TIME = 4.0


class FoodCollectorEnv:
    def __init__(self, num_agents=NUM_AGENTS):
        self.num_agents = num_agents
        self.width = AREA_SIZE
        self.height = AREA_SIZE
        self.agents_pos = np.zeros((num_agents, 2))
        self.agents_rot = np.zeros(num_agents)  # angle in radians
        self.agents_vel = np.zeros((num_agents, 2))
        self.agents_frozen = np.zeros(num_agents, dtype=bool)
        self.agents_frozen_time = np.zeros(num_agents)

        self.good_food = np.zeros((NUM_GOOD_FOOD, 2))
        self.bad_food = np.zeros((NUM_BAD_FOOD, 2))

        self.reset()

    def reset(self):
        for i in range(self.num_agents):
            self.agents_pos[i] = np.random.rand(2) * AREA_SIZE
            self.agents_rot[i] = np.random.rand() * 2 * np.pi
        self.agents_vel.fill(0)
        self.agents_frozen.fill(False)
        self.agents_frozen_time.fill(0)

        for i in range(NUM_GOOD_FOOD):
            self.good_food[i] = np.random.rand(2) * AREA_SIZE
        for i in range(NUM_BAD_FOOD):
            self.bad_food[i] = np.random.rand(2) * AREA_SIZE

        self.steps = 0
        return self._get_all_obs()

    def step(self, actions: List[Tuple[np.ndarray, int]]):
        self.steps += 1
        rewards = np.zeros(self.num_agents)

        # --- Update agents based on actions ---
        is_shooting = np.zeros(self.num_agents, dtype=bool)
        for i in range(self.num_agents):
            if self.agents_frozen[i]:
                if self.steps * 0.03 > self.agents_frozen_time[i] + FROZEN_TIME:
                    self.agents_frozen[i] = False
                continue

            cont_actions, disc_action = actions[i]

            # Action: 3 continuous, 1 discrete
            # cont[0]: forward, cont[1]: right, cont[2]: rotate
            # disc[0]: shoot laser
            forward_move = cont_actions[0] * 2.0
            side_move = cont_actions[1] * 2.0
            rotation = cont_actions[2] * 3.0

            # Update rotation
            self.agents_rot[i] += rotation * 0.1

            # Update velocity
            direction_vec = np.array(
                [np.cos(self.agents_rot[i]), np.sin(self.agents_rot[i])]
            )
            side_vec = np.array(
                [-np.sin(self.agents_rot[i]), np.cos(self.agents_rot[i])]
            )

            force = direction_vec * forward_move + side_vec * side_move
            self.agents_vel[i] += force * 0.1
            self.agents_vel[i] *= 0.95  # damping

            # Update position
            self.agents_pos[i] += self.agents_vel[i]

            # Wall collision
            bounce_factor = -0.5
            if self.agents_pos[i, 0] < AGENT_RADIUS:
                self.agents_pos[i, 0] = AGENT_RADIUS
                self.agents_vel[i, 0] *= bounce_factor
            elif self.agents_pos[i, 0] > self.width - AGENT_RADIUS:
                self.agents_pos[i, 0] = self.width - AGENT_RADIUS
                self.agents_vel[i, 0] *= bounce_factor

            if self.agents_pos[i, 1] < AGENT_RADIUS:
                self.agents_pos[i, 1] = AGENT_RADIUS
                self.agents_vel[i, 1] *= bounce_factor
            elif self.agents_pos[i, 1] > self.height - AGENT_RADIUS:
                self.agents_pos[i, 1] = self.height - AGENT_RADIUS
                self.agents_vel[i, 1] *= bounce_factor

            if disc_action == 1:  # Shoot
                is_shooting[i] = True

        # --- Handle laser shots ---
        for i in range(self.num_agents):
            if is_shooting[i]:
                start_pos = self.agents_pos[i]
                direction = np.array(
                    [np.cos(self.agents_rot[i]), np.sin(self.agents_rot[i])]
                )

                for j in range(self.num_agents):
                    if i == j:
                        continue
                    # Simple segment-circle intersection
                    # This is a simplification; true raycast is more complex
                    target_pos = self.agents_pos[j]
                    vec_to_target = target_pos - start_pos
                    proj = vec_to_target.dot(direction)
                    if 0 < proj < LASER_LENGTH:
                        dist_sq = np.sum(vec_to_target**2) - proj**2
                        if dist_sq < AGENT_RADIUS**2:
                            self.agents_frozen[j] = True
                            self.agents_frozen_time[j] = self.steps * 0.03

        # --- Food collisions ---
        for i in range(self.num_agents):
            # Good food
            for j in range(NUM_GOOD_FOOD):
                if (
                    np.linalg.norm(self.agents_pos[i] - self.good_food[j])
                    < AGENT_RADIUS + FOOD_RADIUS
                ):
                    rewards[i] += 1.0
                    self.good_food[j] = np.random.rand(2) * AREA_SIZE  # respawn
            # Bad food
            for j in range(NUM_BAD_FOOD):
                if (
                    np.linalg.norm(self.agents_pos[i] - self.bad_food[j])
                    < AGENT_RADIUS + FOOD_RADIUS
                ):
                    rewards[i] -= 1.0
                    self.bad_food[j] = np.random.rand(2) * AREA_SIZE  # respawn

        done = self.steps > 3000
        dones = [done] * self.num_agents

        return self._get_all_obs(is_shooting), rewards, dones

    def _get_obs_for_agent(self, agent_idx, is_shooting):
        agent_pos = self.agents_pos[agent_idx]
        agent_rot = self.agents_rot[agent_idx]

        # 1. Local velocity (2)
        # Transform world velocity to agent's local coordinate frame
        world_vel = self.agents_vel[agent_idx]
        cos_rot = np.cos(-agent_rot)
        sin_rot = np.sin(-agent_rot)
        local_vel = np.array(
            [
                world_vel[0] * cos_rot - world_vel[1] * sin_rot,
                world_vel[0] * sin_rot + world_vel[1] * cos_rot,
            ]
        )

        # 2. Frozen & shoot status (2)
        frozen_status = 1.0 if self.agents_frozen[agent_idx] else 0.0
        shoot_status = 1.0 if is_shooting[agent_idx] else 0.0

        # 3. Grid sensor (7x7=49)
        grid_size = 7
        grid_range = 20.0
        grid = np.zeros((grid_size, grid_size))

        # This is a simplified grid sensor. It populates a grid with values
        # indicating objects in range. A real GridSensor is more complex.
        def world_to_grid(pos):
            relative_pos = pos - agent_pos
            # Rotate into agent's reference frame
            x = relative_pos[0] * np.cos(-agent_rot) - relative_pos[1] * np.sin(
                -agent_rot
            )
            y = relative_pos[0] * np.sin(-agent_rot) + relative_pos[1] * np.cos(
                -agent_rot
            )

            if abs(x) > grid_range or abs(y) > grid_range:
                return None

            grid_x = int((x / grid_range * grid_size / 2) + grid_size / 2)
            grid_y = int((y / grid_range * grid_size / 2) + grid_size / 2)

            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                return grid_x, grid_y
            return None

        # Detect other agents
        for i in range(self.num_agents):
            if i == agent_idx:
                continue
            coords = world_to_grid(self.agents_pos[i])
            if coords:
                grid[coords] = 0.25 if self.agents_frozen[i] else 0.5

        # Detect food
        for pos in self.good_food:
            coords = world_to_grid(pos)
            if coords:
                grid[coords] = 1.0
        for pos in self.bad_food:
            coords = world_to_grid(pos)
            if coords:
                grid[coords] = -1.0

        return np.concatenate(
            [local_vel, [frozen_status, shoot_status], grid.flatten()]
        )

    def _get_all_obs(self, is_shooting=None):
        if is_shooting is None:
            is_shooting = np.zeros(self.num_agents, dtype=bool)
        return [self._get_obs_for_agent(i, is_shooting) for i in range(self.num_agents)]

    def get_state_for_viz(self) -> Dict[str, Any]:
        return {
            "agents": [
                {"pos": pos.tolist(), "rot": float(rot), "frozen": bool(f)}
                for pos, rot, f in zip(
                    self.agents_pos, self.agents_rot, self.agents_frozen
                )
            ],
            "good_food": self.good_food.tolist(),
            "bad_food": self.bad_food.tolist(),
            "bounds": [self.width, self.height],
        }
