import gymnasium as gym
from gymnasium import spaces

import numpy as np
import traci
import os
import sys
import random


class LaneChangeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, sumo_cfg_file, mode="continuous", use_gui=False, simulation_steps=1000):
        super(LaneChangeEnv, self).__init__()

        self.sumo_cfg_file = sumo_cfg_file
        self.mode = mode
        self.use_gui = use_gui
        self.simulation_steps = simulation_steps
        self.ego_id = "ego"
        self.track_len = 1000

        # Action Space: 0: Keep, 1: Change Left, 2: Change Right
        self.action_space = spaces.Discrete(3)

        # Observation Space
        if self.mode == "continuous":
            # [ego_lane, ego_speed, dist_to_leader, speed_leader, dist_to_follower, speed_follower, ... for other lane]
            # Simplified: [ego_lane_index, dist_to_closest_obstacle_fwd, dist_to_closest_obstacle_bwd, lateral_pos]
            # We normalize these values roughly
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        else:
            # Discrete State Space for Q-Learning
            # We discretize the surroundings into a grid or specific states
            # Example: (Lane_ID, Is_Obstacle_Front_Close, Is_Obstacle_Left_Close, Is_Obstacle_Right_Close)
            # Let's say: Lane (0,1), Front (0=Safe, 1=Close), Left (0=Safe, 1=Close), Right (0=Safe, 1=Close)
            # Size: 2 * 2 * 2 * 2 = 16 states.
            # We return an integer index or a tuple. Gym expects a Box or Discrete.
            # For Q-Learning implementation, we might handle the conversion from observation to state index outside or here.
            # Let's return a Box here for consistency, and the Q-agent will discretize it.
            self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int32)

        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.label = str(random.randint(0, 10000))  # Unique label for traci connection
        self.running = False
        self.sumo_control = False

    def set_sumo_control(self, enable=True):
        self.sumo_control = enable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.running:
            traci.close()
            self.running = False

        # Start SUMO
        sumo_cmd = [self.sumo_binary, "-c", self.sumo_cfg_file, "--start", "--quit-on-end"]
        if not self.use_gui:
            sumo_cmd.append("--no-step-log")
            sumo_cmd.append("--no-warnings")
            # Redirect log to null to keep console clean
            log_file = "NUL" if os.name == "nt" else "/dev/null"
            sumo_cmd.extend(["--log", log_file])

        # Forcefully suppress stdout/stderr during startup
        if not self.use_gui:
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:
                    traci.start(sumo_cmd, label=self.label)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
        else:
            traci.start(sumo_cmd, label=self.label, stdout=None)

        self.running = True

        # Setup Traffic
        self._setup_traffic()

        # Step once to initialize
        traci.simulationStep()

        return self._get_obs(), {}

    def _setup_traffic(self):
        # Add Ego Vehicle
        # Ensure route exists (defined in rou.xml)
        try:
            traci.vehicle.add(self.ego_id, "r0", typeID="ego", departPos="0", departLane="0")
            traci.vehicle.setSpeedMode(self.ego_id, 31)  # All checks on

            if self.sumo_control:
                # Enable default lane changing (1621)
                traci.vehicle.setLaneChangeMode(self.ego_id, 1621)
            else:
                # Disable all autonomous lane changing
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        except traci.exceptions.TraCIException:
            pass  # Vehicle might already exist if we didn't close properly

        # Add Obstacles
        # Randomly place 5-10 obstacles
        num_obstacles = random.randint(5, 10)
        placed_obstacles = []  # List of (pos, lane)

        for i in range(num_obstacles):
            obs_id = f"obs_{i}"

            # Try to find a valid position
            attempts = 0
            while attempts < 100:
                lane = random.randint(0, 1)
                pos = random.randint(50, 900)  # Avoid start and end

                valid = True
                for p, l in placed_obstacles:
                    # Check 1: Overlap in same lane (collision)
                    if l == lane and abs(p - pos) < 20:
                        valid = False
                        break
                    # Check 2: Blockage in other lane (wall)
                    # If obstacles are in different lanes, ensure they aren't too close
                    # to allow the agent to weave through.
                    if l != lane and abs(p - pos) < 30:  # 30m gap to avoid "gate" effect
                        valid = False
                        break

                if valid:
                    break
                attempts += 1

            if attempts < 100:
                try:
                    traci.vehicle.add(
                        obs_id, "r0", typeID="obstacle", departPos=str(pos), departLane=str(lane)
                    )
                    traci.vehicle.setSpeed(obs_id, 0)  # Motionless
                    traci.vehicle.setColor(obs_id, (255, 0, 0, 255))
                    traci.vehicle.setLaneChangeMode(obs_id, 0)  # Disable lane changing
                    placed_obstacles.append((pos, lane))
                except traci.exceptions.TraCIException:
                    pass

    def step(self, action):
        if not self.running:
            return self._get_obs(), 0, True, False, {}

        # Check if ego exists
        if self.ego_id not in traci.vehicle.getIDList():
            return self._get_obs(), -100, True, False, {}

        # Apply Action
        # 0: Keep, 1: Left, 2: Right
        if not self.sumo_control:
            current_lane = traci.vehicle.getLaneIndex(self.ego_id)

            if action == 1:  # Left
                if current_lane < 1:  # Assuming 2 lanes (0, 1)
                    traci.vehicle.changeLane(self.ego_id, current_lane + 1, duration=1)
            elif action == 2:  # Right
                if current_lane > 0:
                    traci.vehicle.changeLane(self.ego_id, current_lane - 1, duration=1)

        # Simulate
        traci.simulationStep()

        # Check if ego still exists after step
        if self.ego_id not in traci.vehicle.getIDList():
            # Vehicle crashed or finished
            # We can try to infer if it was a crash based on collision list
            terminated = True
            reward = -50  # Penalty for disappearing (likely crash)
            # Return dummy observation
            if self.mode == "continuous":
                obs = np.zeros(6, dtype=np.float32)
            else:
                obs = np.zeros(4, dtype=np.int32)

            traci.close()
            self.running = False
            return obs, reward, terminated, False, {}

        # Get State & Reward
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_collision() or self._check_goal()
        truncated = traci.simulation.getTime() > self.simulation_steps

        info = {}

        if terminated or truncated:
            traci.close()
            self.running = False

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        try:
            ego_pos = traci.vehicle.getPosition(self.ego_id)
            ego_lane = traci.vehicle.getLaneIndex(self.ego_id)

            # Find closest obstacles
            # We need to scan all vehicles
            vehicles = traci.vehicle.getIDList()
            min_dist_front = 1000
            min_dist_back = 1000

            # For simplified observation, let's look at:
            # 1. Ego Lane Index (0 or 1)
            # 2. Distance to closest object in CURRENT lane (front)
            # 3. Distance to closest object in LEFT lane (front/back) - simplified to 'is safe to merge'
            # 4. Distance to closest object in RIGHT lane (front/back)

            # Let's construct a 6-dim vector for continuous:
            # [lane_idx, dist_front_curr, dist_back_curr, dist_front_left, dist_front_right, dist_back_left, dist_back_right]
            # Wait, let's stick to the plan.

            # Get all obstacles
            obstacles = [v for v in vehicles if v != self.ego_id]

            # Helper to find distance
            def get_lane_dists(lane_idx):
                dists_front = []
                dists_back = []
                for obs in obstacles:
                    if traci.vehicle.getLaneIndex(obs) == lane_idx:
                        obs_pos = traci.vehicle.getPosition(obs)
                        dist = obs_pos[0] - ego_pos[0]  # Relative X
                        if dist > 0:
                            dists_front.append(dist)
                        else:
                            dists_back.append(abs(dist))
                return min(dists_front) if dists_front else 200, min(dists_back) if dists_back else 200

            d_front_curr, d_back_curr = get_lane_dists(ego_lane)
            d_front_left, d_back_left = (
                get_lane_dists(ego_lane + 1) if ego_lane < 1 else (0, 0)
            )  # 0 indicates wall/invalid
            d_front_right, d_back_right = get_lane_dists(ego_lane - 1) if ego_lane > 0 else (0, 0)

            if self.mode == "continuous":
                # Normalize somewhat
                return np.array(
                    [
                        ego_lane,
                        d_front_curr / 200.0,
                        d_back_curr / 200.0,
                        d_front_left / 200.0,
                        d_back_left / 200.0,
                        d_front_right / 200.0,
                    ],
                    dtype=np.float32,
                )
            else:
                # Discrete Mode for Q-Learning
                # State: [Lane, Safe_Front, Safe_Left, Safe_Right]
                # Safe if dist > 20m (example)
                safe_dist = 30
                is_safe_front = 1 if d_front_curr > safe_dist else 0

                # For side lanes, we need both front and back to be safe to merge
                is_safe_left = 1 if (ego_lane < 1 and d_front_left > safe_dist and d_back_left > 10) else 0
                is_safe_right = 1 if (ego_lane > 0 and d_front_right > safe_dist and d_back_right > 10) else 0

                return np.array([ego_lane, is_safe_front, is_safe_left, is_safe_right], dtype=np.int32)

        except traci.exceptions.TraCIException:
            return np.zeros(6 if self.mode == "continuous" else 4, dtype=np.float32)

    def _compute_reward(self, action):
        # ----------------------
        # Constants (Tune these)
        # ----------------------
        MAX_SPEED = 30.0  # e.g., 30 m/s (~108 km/h)
        COLLISION_PENALTY = -50.0
        LANE_CHANGE_COST = -0.5  # Small penalty to discourage jitter
        HEADWAY_COST = -1.0
        SAFE_HEADWAY = 2.0  # seconds (Two-second rule)

        reward = 0.0

        # ----------------------
        # 1. High Speed Reward (The Engine)
        # ----------------------
        # We reward the agent for driving near max speed.
        # If stuck behind a car, this drops, naturally motivating a lane change.
        v_ego = traci.vehicle.getSpeed(self.ego_id)
        r_speed = v_ego / MAX_SPEED  # Normalized 0.0 to 1.0
        reward += r_speed

        # ----------------------
        # 2. Stability Penalty (The Stabilizer)
        # ----------------------
        # Penalize every lane change. This stops the "zigzag farming".
        # The agent will only change lanes if the potential speed gain > 0.5
        if action != 0:  # Assuming 0 is Keep Lane
            reward += LANE_CHANGE_COST

        # ----------------------
        # 3. Safety/Headway (The Brakes)
        # ----------------------
        # Instead of fixed meters, use Time Headway (Distance / Speed)
        # This scales with how fast you are driving.

        # Get distance to closest leader (your existing logic is fine here)
        # Note: traci.vehicle.getLeader returns (vehicle_id, dist) or None
        leader_info = traci.vehicle.getLeader(self.ego_id)
        if leader_info:
            leader_dist = leader_info[1]
            # Calculate time headway (avoid division by zero)
            time_headway = leader_dist / (v_ego + 0.001)

            # If we are closer than 2 seconds, apply penalty
            if time_headway < SAFE_HEADWAY:
                # Penalty increases as we get closer (0 to -1.0)
                # Formula: -(2.0 - current) / 2.0
                r_headway = -(SAFE_HEADWAY - time_headway) / SAFE_HEADWAY
                reward += r_headway

        # ----------------------
        # 4. Collision (Crucial)
        # ----------------------
        # If a collision occurred in this step (check traci collision list)
        # usually handled in 'step' but can be added here if 'done' flag isn't enough
        # reward += COLLISION_PENALTY

        return reward

    def _check_collision(self):
        # SUMO reports collisions
        colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in colliding_vehicles:
            return True
        return False

    def _check_goal(self):
        try:
            p = traci.vehicle.getPosition(self.ego_id)
            if p[0] > 950:  # End of track
                return True
        except:
            pass
        return False

    def close(self):
        if self.running:
            traci.close()
            self.running = False
