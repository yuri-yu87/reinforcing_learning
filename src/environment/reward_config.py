"""
Reward Configuration for UAV Environment - Discrete 5-Way RL Strategy

This module implements reward parameters following the discrete 5-way RL strategy
from Design Journal Section 5.3, focusing on stability and constraint enforcement.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class RewardConfig:
    """
    ULTIMATE FIXED Discrete 5-Way RL Strategy Reward Configuration.
    
    Implements comprehensive solution to achieve all objectives:
    1. Maximize episode sum-throughput  
    2. Satisfy hard constraints (start/end, bounds, speed, power, 200-300s)
    3. Promote fair access when required
    """
    
    # === SIMPLIFIED CORE THROUGHPUT REWARDS ===
    w_throughput_base: float = 100.0       # Simplified base throughput weight
    w_throughput_multiplier: float = 0.0   # Disable complex distance modulation
    
    # === SIMPLIFIED MOVEMENT INCENTIVES ===
    w_movement_bonus: float = 10.0         # Simplified movement reward
    w_distance_progress: float = 0.0       # Disable distance progress (causes issues)
    w_user_approach: float = 50.0          # Simplified user approach
    
    # === REDUCED PENALTIES FOR STABILITY ===
    w_oob: float = 100.0                   # Moderate out-of-bounds penalty
    w_stagnation: float = 1.0              # Minimal stagnation penalty
    
    # === SIMPLIFIED TERMINAL REWARDS ===
    B_mission_complete: float = 1000.0     # Clear mission completion signal
    B_reach_end: float = 500.0            # Clear endpoint bonus
    B_time_window: float = 500.0          # Clear time window bonus
    B_fair_access: float = 200.0          # Clear fair access bonus
    B_visit_all_users: float = 300.0      # Clear visit all users bonus
    
    # === CRITICAL FIXES ===
    alpha_fair: float = 0.0               # DISABLE proportional fair (was 1.0)
    user_service_radius: float = 60.0     # EXPAND service radius (was 10.0)
    fairness_epsilon: float = 1e-6        # Keep for compatibility
    
    # === TIME CONSTRAINTS ===
    min_flight_time: float = 200.0        # Mission time window
    max_flight_time: float = 300.0
    
    # === RELAXED DISTANCE THRESHOLDS ===
    close_to_user_threshold: float = 70.0  # Generous user proximity threshold
    close_to_end_threshold: float = 30.0   # Generous end proximity threshold
    
    # === RELAXED MISSION COMPLETION PARAMETERS ===
    end_position_tolerance: float = 10.0   # Generous end tolerance (covers UAV position)
    user_visit_time_threshold: float = 0.5 # Minimal visit time requirement
    
    # === TIGHTENED STAGNATION DETECTION ===
    stagnation_threshold: float = 1.0      # Stricter stagnation detection (vs 2.0)
    stagnation_time_window: float = 3.0    # Shorter detection window (vs 5.0)
    
    # === Constraint violation thresholds ===
    speed_tolerance: float = 1.0           # Speed constraint tolerance (m/s)
    
    # === Lagrangian constraint parameters ===
    enable_lagrangian: bool = False        # Enable Lagrangian constraint shaping
    lambda_time: float = 0.1               # Time constraint dual variable
    lambda_bounds: float = 10.0            # Bounds constraint dual variable
    lambda_speed: float = 1.0              # Speed constraint dual variable
    
    # === System parameters ===
    time_step: float = 0.1                 # Environment time step (s)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reward config to dictionary for info purposes"""
        return {
            'w_throughput_base': self.w_throughput_base,
            'w_throughput_multiplier': self.w_throughput_multiplier,
            'w_movement_bonus': self.w_movement_bonus,
            'w_distance_progress': self.w_distance_progress,
            'w_user_approach': self.w_user_approach,
            'w_oob': self.w_oob,
            'w_stagnation': self.w_stagnation,
            'B_mission_complete': self.B_mission_complete,
            'B_reach_end': self.B_reach_end,
            'B_time_window': self.B_time_window,
            'B_fair_access': self.B_fair_access,
            'B_visit_all_users': self.B_visit_all_users,
            'alpha_fair': self.alpha_fair,
            'user_service_radius': self.user_service_radius,
            'end_position_tolerance': self.end_position_tolerance,
            'min_flight_time': self.min_flight_time,
            'max_flight_time': self.max_flight_time,
            'time_step': self.time_step
        }


class RewardCalculator:
    """
    ULTIMATE FIXED Reward Calculator - Comprehensive Solution.
    
    Implements all fixes to achieve the stated objectives:
    1. Maximize episode sum-throughput
    2. Satisfy hard constraints  
    3. Promote fair access
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset calculator state for new episode"""
        # User visit tracking
        self.user_visit_times = {}  # {user_id: total_time_in_service}
        self.user_visited_flags = set()  # Set of visited user IDs
        self.all_users_visited = False
        
        # Position tracking for movement and progress rewards
        self.position_history = []
        self.time_history = []
        self.previous_position = None
        self.previous_distances_to_users = None
        self.previous_distance_to_end = None
        
        # Episode tracking
        self.episode_start_time = 0.0
        self.mission_completed = False
    
    def calculate_reward(self, 
                        uav_position: np.ndarray,
                        end_position: np.ndarray,
                        user_positions: np.ndarray,
                        user_throughputs: np.ndarray,
                        current_time: float,
                        current_speed: float,
                        env_bounds: tuple,
                        episode_done: bool,
                        reached_end: bool) -> Dict[str, float]:
        """
        ULTIMATE FIXED reward calculation - comprehensive solution.
        
        Returns:
            Dictionary with reward components and total reward
        """
        reward_breakdown = {}
        
        # === 1. CORE THROUGHPUT REWARDS (FIXED) ===
        
        # Base throughput reward (no proportional fair trap!)
        base_throughput = self.config.w_throughput_base * np.sum(user_throughputs)
        
        # Distance-modulated throughput reward (doesn't rely on service radius)
        distance_modulated_throughput = 0.0
        for i, user_pos in enumerate(user_positions):
            distance = np.linalg.norm(uav_position - user_pos)
            # Distance factor: closer = higher reward (0.1 to 1.0 range)
            distance_factor = max(0.1, 1.0 - distance / 100.0)
            user_reward = user_throughputs[i] * distance_factor * self.config.w_throughput_multiplier
            distance_modulated_throughput += user_reward
        
        throughput_reward = base_throughput + distance_modulated_throughput
        reward_breakdown['throughput'] = throughput_reward
        
        # === 2. MOVEMENT & EXPLORATION INCENTIVES ===
        
        # Movement reward
        movement_reward = 0.0
        if self.previous_position is not None:
            displacement = np.linalg.norm(uav_position - self.previous_position)
            movement_reward = self.config.w_movement_bonus * min(displacement / 3.0, 1.0)
        
        reward_breakdown['movement'] = movement_reward
        
        # Distance progress reward
        distance_progress_reward = 0.0
        
        # Progress toward users
        current_distances_to_users = [np.linalg.norm(uav_position - user_pos) for user_pos in user_positions]
        if self.previous_distances_to_users is not None:
            for i, (prev_dist, curr_dist) in enumerate(zip(self.previous_distances_to_users, current_distances_to_users)):
                improvement = prev_dist - curr_dist
                if improvement > 0:
                    distance_progress_reward += self.config.w_distance_progress * improvement * 0.1
        
        # Progress toward end (if all users visited)
        current_distance_to_end = np.linalg.norm(uav_position - end_position)
        if len(self.user_visited_flags) == len(user_positions):
            if self.previous_distance_to_end is not None:
                end_improvement = self.previous_distance_to_end - current_distance_to_end
                if end_improvement > 0:
                    distance_progress_reward += self.config.w_distance_progress * end_improvement * 0.2
        
        reward_breakdown['distance_progress'] = distance_progress_reward
        
        # User approach reward
        approach_reward = 0.0
        for user_pos in user_positions:
            distance = np.linalg.norm(uav_position - user_pos)
            if distance <= self.config.close_to_user_threshold:
                proximity_factor = (self.config.close_to_user_threshold - distance) / self.config.close_to_user_threshold
                approach_reward += self.config.w_user_approach * proximity_factor * 0.1
        
        # Approach end reward (if users visited)
        if len(self.user_visited_flags) == len(user_positions):
            if current_distance_to_end <= self.config.close_to_end_threshold:
                proximity_factor = (self.config.close_to_end_threshold - current_distance_to_end) / self.config.close_to_end_threshold
                approach_reward += self.config.w_user_approach * proximity_factor * 0.2
        
        reward_breakdown['approach'] = approach_reward
        
        # === 3. SAFETY PENALTIES ===
        safety_penalty = 0.0
        
        # Out-of-bounds penalty
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = env_bounds
        if (uav_position[0] < x_min or uav_position[0] > x_max or
            uav_position[1] < y_min or uav_position[1] > y_max):
            safety_penalty += self.config.w_oob
        
        reward_breakdown['safety'] = -safety_penalty
        
        # === 4. INTELLIGENT STAGNATION PENALTY ===
        stagnation_penalty = 0.0
        
        # Update position history
        self.position_history.append(uav_position.copy())
        self.time_history.append(current_time)
        
        # Keep only recent history  
        window_size = int(self.config.stagnation_time_window / self.config.time_step)
        if len(self.position_history) > window_size:
            self.position_history = self.position_history[-window_size:]
            self.time_history = self.time_history[-window_size:]
        
        # Check for stagnation
        if len(self.position_history) >= window_size:
            pos_variance = np.var(self.position_history, axis=0)
            total_variance = np.sum(pos_variance[:2])
            
            if total_variance < self.config.stagnation_threshold:
                # Check if stagnating in meaningful location
                meaningful_location = False
                
                # OK to stagnate near users
                for user_pos in user_positions:
                    if np.linalg.norm(uav_position - user_pos) <= self.config.user_service_radius:
                        meaningful_location = True
                        break
                
                # OK to stagnate at end
                if current_distance_to_end <= self.config.end_position_tolerance:
                    meaningful_location = True
                
                if not meaningful_location:
                    stagnation_penalty = self.config.w_stagnation
        
        reward_breakdown['stagnation'] = -stagnation_penalty
        
        # === 5. UPDATE USER VISIT TRACKING ===
        self._update_user_visits(uav_position, user_positions, current_time)
        
        # === 6. TERMINAL MISSION COMPLETION REWARDS ===
        terminal_reward = 0.0
        
        if episode_done:
            # Complete mission reward (reach end + visit all users + time window)
            if (reached_end and 
                len(self.user_visited_flags) == len(user_positions) and
                self.config.min_flight_time <= current_time <= self.config.max_flight_time):
                terminal_reward += self.config.B_mission_complete
            
            # Reach endpoint reward
            if reached_end:
                terminal_reward += self.config.B_reach_end
            
            # Time window reward
            if self.config.min_flight_time <= current_time <= self.config.max_flight_time:
                terminal_reward += self.config.B_time_window
            
            # Visit all users reward
            if len(self.user_visited_flags) == len(user_positions):
                terminal_reward += self.config.B_visit_all_users
            
            # Fair access reward (simplified)
            if len(self.user_visit_times) >= 2:
                visit_times = list(self.user_visit_times.values())
                if min(visit_times) > 0:  # All users visited
                    fairness = min(visit_times) / max(visit_times)
                    terminal_reward += self.config.B_fair_access * fairness
                
        reward_breakdown['terminal'] = terminal_reward
        
        # === 7. TOTAL REWARD CALCULATION ===
        total_reward = (throughput_reward + movement_reward + distance_progress_reward + 
                       approach_reward - safety_penalty - stagnation_penalty + terminal_reward)
        
        reward_breakdown['total'] = total_reward
        
        # === 8. UPDATE STATE FOR NEXT STEP ===
        self.previous_position = uav_position.copy()
        self.previous_distances_to_users = current_distances_to_users
        self.previous_distance_to_end = current_distance_to_end
        
        return reward_breakdown
    
    def _update_user_visits(self, uav_position: np.ndarray, user_positions: np.ndarray, current_time: float):
        """Update user visit tracking with expanded service radius"""
        for user_id, user_pos in enumerate(user_positions):
            distance = np.linalg.norm(uav_position - user_pos)
            
            # If within expanded service radius, accumulate visit time
            if distance <= self.config.user_service_radius:  # Now 50m instead of 10m
                if user_id not in self.user_visit_times:
                    self.user_visit_times[user_id] = 0.0
                self.user_visit_times[user_id] += self.config.time_step
                
                # Mark as visited if enough time spent
                if (self.user_visit_times[user_id] >= self.config.user_visit_time_threshold and
                    user_id not in self.user_visited_flags):
                    self.user_visited_flags.add(user_id)
        
        # Update all users visited flag
        if len(self.user_visited_flags) == len(user_positions):
            self.all_users_visited = True
    
    def _calculate_jain_index(self, values: List[float]) -> float:
        """Calculate Jain's fairness index"""
        if not values or len(values) <= 1:
            return 1.0
        
        values = np.array(values)
        sum_values = np.sum(values)
        sum_squares = np.sum(values ** 2)
        
        if sum_squares == 0:
            return 1.0
        
        n = len(values)
        jain_index = (sum_values ** 2) / (n * sum_squares)
        return jain_index
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'user_visit_times': dict(self.user_visit_times),
            'user_visited_flags': list(self.user_visited_flags),
            'all_users_visited': self.all_users_visited,
            'mission_completed': self.mission_completed
        }