"""
Baseline agent implementations for comparison.
Includes deterministic strategies as baselines for RL algorithms.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import gymnasium as gym

from .base_agent import BaseAgent


class BaselineAgent(BaseAgent):
    """
    Baseline agent implementing deterministic strategies.
    
    This agent serves as a baseline for comparing RL algorithm performance.
    """
    
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 strategy: str = 'straight_line',
                 beamforming_type: str = 'optimized',  # 'optimized' or 'random'
                 **kwargs):
        """
        Initialize baseline agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            strategy: Strategy to use ('straight_line', 'greedy', 'circular', 'random')
            beamforming_type: Beamforming type ('optimized' or 'random')
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space, **kwargs)
        
        self.strategy = strategy
        self.beamforming_type = beamforming_type
        self.start_position = np.array([0, 0, 50])  # Default start
        self.end_position = np.array([80, 80, 50])   # Default end
        self.current_position = self.start_position.copy()
        self.target_position = self.end_position.copy()
        
        # Add method to update target from environment
        self.environment_target = None
        
        # Strategy-specific parameters
        self.circular_center = None
        self.circular_radius = 20.0
        self.greedy_target_user = 0
        
        # Enhanced greedy strategy parameters
        self.visited_users = set()  # Track visited users
        self.current_target_user = None  # Current target user
        self.user_visit_threshold = 100.0  # Distance threshold to consider user "visited" (increased for long-range users)
        self.mission_phase = 'user_service'  # 'user_service' or 'go_to_target'
        self.user_service_time = {}  # Track time spent near each user
    
    def set_target_position(self, target_position: np.ndarray):
        """Set target position from environment."""
        self.environment_target = target_position.copy()
    
    def set_beamforming_type(self, beamforming_type: str):
        """Set beamforming type for the agent."""
        if beamforming_type not in ['optimized', 'random']:
            raise ValueError(f"Unknown beamforming type: {beamforming_type}")
        self.beamforming_type = beamforming_type
    
    def reset(self):
        """Reset agent state for new episode."""
        super().reset()
        # Reset greedy strategy state
        self.visited_users = set()
        self.current_target_user = None
        self.mission_phase = 'user_service'
        self.user_service_time = {}
        # Reset circular strategy state
        self.circular_center = None
        # Reset debug state
        self._debug_visited_users = set()
        
    def select_action(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action based on the chosen strategy.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic action selection (always True for baseline)
            
        Returns:
            Selected discrete action (0=East, 1=South, 2=West, 3=North, 4=Hover)
        """
        # Extract current UAV position from observation
        # Observation format: [uav_pos(3), end_pos(3), remaining_time(1), signal_quality(num_users), throughput_history(5)]
        current_position = observation[:3]
        end_position = observation[3:6]
        
        # User positions should be passed from the environment
        # For now, use a placeholder - this will be fixed in the test
        user_positions = getattr(self, '_current_user_positions', np.array([[0, 0, 0], [0, 0, 0]]))
        
        # Update current position and target
        self.current_position = current_position
        self.target_position = end_position
        
        if self.strategy == 'straight_line':
            action = self._straight_line_strategy_discrete()
        elif self.strategy == 'greedy':
            action = self._greedy_strategy_discrete(user_positions)
        elif self.strategy == 'circular':
            action = self._circular_strategy_discrete(user_positions)
        elif self.strategy == 'random':
            action = self._random_strategy_discrete()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Update step count
        self.total_steps += 1
        
        return action
    
    def _straight_line_strategy_discrete(self) -> int:
        """
        Discrete straight line strategy: move toward target using 4 cardinal directions.
        
        Returns:
            Discrete action (0=East, 1=South, 2=West, 3=North, 4=Hover)
        """
        # Calculate direction to target
        if self.environment_target is not None:
            target_position = self.environment_target
        else:
            target_position = self.target_position
        
        direction = target_position - self.current_position
        distance = np.linalg.norm(direction[:2])  # Only consider X-Y plane
        
        # If very close to target, hover
        if distance < 2.0:
            return 4  # Hover
        
        # Determine primary direction based on largest component
        dx, dy = direction[0], direction[1]
        
        if abs(dx) > abs(dy):
            # Move primarily in X direction
            if dx > 0:
                return 0  # East (+X)
            else:
                return 2  # West (-X)
        else:
            # Move primarily in Y direction
            if dy > 0:
                return 3  # North (+Y)
            else:
                return 1  # South (-Y)
    
    def _greedy_strategy_discrete(self, user_positions: np.ndarray) -> int:
        """
        Discrete greedy strategy: visit nearest user first, then go to target.
        
        Args:
            user_positions: Array of user positions
            
        Returns:
            Discrete action
        """
        # Simple greedy: find nearest user and go there, then target
        user_distances = [np.linalg.norm(self.current_position - user_pos) 
                         for user_pos in user_positions]
        nearest_user_idx = np.argmin(user_distances)
        nearest_user_pos = user_positions[nearest_user_idx]
        
        # If close to nearest user, go to target
        if user_distances[nearest_user_idx] < 10.0:
            target_direction = self.target_position - self.current_position
        else:
            target_direction = nearest_user_pos - self.current_position
        
        distance = np.linalg.norm(target_direction[:2])
        if distance < 2.0:
            return 4  # Hover
        
        # Choose direction
        dx, dy = target_direction[0], target_direction[1]
        if abs(dx) > abs(dy):
            return 0 if dx > 0 else 2  # East or West
        else:
            return 3 if dy > 0 else 1  # North or South
    
    def _circular_strategy_discrete(self, user_positions: np.ndarray) -> int:
        """
        Discrete circular strategy: simplified circular motion.
        
        Args:
            user_positions: Array of user positions
            
        Returns:
            Discrete action
        """
        # Simplified circular: rotate between directions
        cycle_actions = [0, 3, 2, 1]  # East, North, West, South
        cycle_position = self.total_steps % len(cycle_actions)
        return cycle_actions[cycle_position]
    
    def _random_strategy_discrete(self) -> int:
        """
        Discrete random strategy: randomly select from 5 actions.
        
        Returns:
            Discrete action
        """
        return np.random.randint(0, 5)
    
    def _straight_line_strategy(self) -> np.ndarray:
        """
        Enhanced precise straight line strategy: move directly from start to end with high precision.
        
        Returns:
            Action array [dx, dy, speed, hover]
        """
        # Calculate direction to target - use environment target if available
        if self.environment_target is not None:
            target_position = self.environment_target
        else:
            target_position = self.target_position
        
        direction = target_position - self.current_position
        distance = np.linalg.norm(direction)
        
        # Enhanced precision control with multiple distance thresholds
        if distance < 0.1:  # Extremely close to target - hover for maximum precision
            return np.array([0.0, 0.0, 10.0, 1.0])  # Hover at target
        elif distance < 0.5:  # Very close - ultra slow approach
            normalized_direction = direction / distance
            speed = 10.0  # Minimum speed for ultra precise approach
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
        elif distance < 1.0:  # Very close - extremely slow approach
            normalized_direction = direction / distance
            speed = 10.0  # Minimum speed for maximum precision
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
        elif distance < 2.0:  # Close - ultra slow approach
            normalized_direction = direction / distance
            speed = min(12.0, max(10.0, distance * 3.0))  # Very slow speed
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
        elif distance < 5.0:  # Close - slow and careful approach
            normalized_direction = direction / distance
            speed = min(15.0, max(10.0, distance * 2.0))  # Adaptive slow speed
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
        elif distance < 15.0:  # Medium distance - moderate speed
            normalized_direction = direction / distance
            speed = min(25.0, max(10.0, distance * 1.5))  # Moderate speed
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
        else:  # Far from target - normal navigation
            if distance > 0:
                # Normalize direction
                normalized_direction = direction / distance
                # Adaptive speed: faster when far, slower when approaching
                speed = min(30.0, max(10.0, 10.0 + distance * 0.2))
            else:
                # Fallback (shouldn't happen)
                normalized_direction = np.array([0.0, 0.0, 0.0])
                speed = 15.0
        
            return np.array([normalized_direction[0], normalized_direction[1], speed, 0.0])
    
    def _greedy_strategy(self, user_positions: np.ndarray) -> np.ndarray:
        """
        Enhanced greedy strategy: Sequential user visitation then target navigation.
        Strategy: Visit User1 -> Visit User2 -> Go to Target (exact position)
        
        Args:
            user_positions: Array of user positions
            
        Returns:
            Action array [dx, dy, speed, hover]
        """
        if len(user_positions) == 0:
            return self._straight_line_strategy()
        
        # Get target position
        if self.environment_target is not None:
            target_position = self.environment_target
        else:
            target_position = self.target_position
        
        # Update visited users based on current position
        self._update_visited_users(user_positions)
        
        # Determine current mission phase and target
        current_target, action_type = self._determine_current_target(user_positions, target_position)
        
        # Calculate direction to current target
        direction = current_target - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance < 1.0:  # Very close to target
            normalized_direction = np.array([0.0, 0.0, 0.0])
            speed = 10.0  # Fixed: minimum speed should be 10
            hover = 1.0 if action_type == 'user_service' else 0.0
        else:
            normalized_direction = direction / distance
            
            # Set speed and hover based on action type and distance
            if action_type == 'user_service':
                # Going to or servicing a user
                if distance < self.user_visit_threshold:
                    # Near user - hover and provide service
                    speed = min(15.0, max(10.0, distance * 0.3))  # Fixed: min speed should be 10
                    hover = 1.0
                    # Track service time
                    user_idx = self.current_target_user
                    if user_idx not in self.user_service_time:
                        self.user_service_time[user_idx] = 0
                    self.user_service_time[user_idx] += 1
                else:
                    # Moving toward user - use higher speed for distant users
                    speed = min(30.0, max(10.0, distance * 0.3))  # Fixed: min speed should be 10
                    hover = 0.0
            
            elif action_type == 'go_to_target':
                # Going to final target - no hovering, direct movement
                if distance < 1.5:
                    # Very close to target - slow for final precision
                    speed = min(10.0, max(10.0, distance * 2.0))  # Fixed: min speed should be 10
                    hover = 0.0
                elif distance < 5.0:
                    # Close to target - slow down for precision
                    speed = min(15.0, max(10.0, distance * 1.2))  # Fixed: min speed should be 10
                    hover = 0.0
                elif distance < 15.0:
                    # Moderate distance - controlled approach
                    speed = min(20.0, max(10.0, distance * 0.8))  # Fixed: min speed should be 10
                    hover = 0.0
                else:
                    # Moving toward target - high speed for navigation
                    speed = min(30.0, max(10.0, distance * 0.4))  # Fixed: min speed should be 10
                    hover = 0.0
            
            else:  # fallback
                speed = 15.0  # Within valid range [10, 30]
                hover = 0.0
        
        return np.array([normalized_direction[0], normalized_direction[1], speed, hover])
    
    def _update_visited_users(self, user_positions: np.ndarray):
        """Update the set of visited users based on current position."""
        for i, user_pos in enumerate(user_positions):
            # Calculate 2D distance (X-Y plane only) since users are on ground (Z=0) and UAV is at Z=50
            distance_2d = np.linalg.norm(self.current_position[:2] - user_pos[:2])
            if distance_2d < self.user_visit_threshold:
                self.visited_users.add(i)
                # Initialize service time if not exists
                if i not in self.user_service_time:
                    self.user_service_time[i] = 0
                # Debug: Print when user is visited for the first time
                if i not in getattr(self, '_debug_visited_users', set()):
                    print(f"  🎯 用户{i+1}首次访问! 距离: {distance_2d:.1f}m < {self.user_visit_threshold}m")
                    if not hasattr(self, '_debug_visited_users'):
                        self._debug_visited_users = set()
                    self._debug_visited_users.add(i)
    
    def _determine_current_target(self, user_positions: np.ndarray, target_position: np.ndarray):
        """
        Determine current target based on mission phase and visited users.
        
        Returns:
            tuple: (target_position, action_type)
        """
        # Check if all users have been sufficiently served
        all_users_served = True
        for i in range(len(user_positions)):
            if i not in self.visited_users or self.user_service_time.get(i, 0) < 5:  # Reduced from 10 to 5
                all_users_served = False
                break
        
        if all_users_served:
            # All users served - go directly to target
            self.mission_phase = 'go_to_target'
            return target_position, 'go_to_target'
        
        # Find next user to visit
        # Priority: unvisited users first, then underserved users
        unvisited_users = []
        underserved_users = []
        
        for i in range(len(user_positions)):
            if i not in self.visited_users:
                unvisited_users.append(i)
            elif self.user_service_time.get(i, 0) < 5:  # Updated to match the threshold
                underserved_users.append(i)
        
        if unvisited_users:
            # Choose closest unvisited user (2D distance)
            target_user_idx = min(unvisited_users, 
                                key=lambda i: np.linalg.norm(self.current_position[:2] - user_positions[i][:2]))
        elif underserved_users:
            # Choose closest underserved user (2D distance)
            target_user_idx = min(underserved_users,
                                key=lambda i: np.linalg.norm(self.current_position[:2] - user_positions[i][:2]))
        else:
            # Fallback - go to target
            self.mission_phase = 'go_to_target'
            return target_position, 'go_to_target'
        
        self.current_target_user = target_user_idx
        self.mission_phase = 'user_service'
        return user_positions[target_user_idx], 'user_service'
    
    def _circular_strategy(self, user_positions: np.ndarray) -> np.ndarray:
        """
        Improved circular strategy: move in a path that serves users while progressing toward target.
        
        Args:
            user_positions: Array of user positions
            
        Returns:
            Action array [dx, dy, speed, hover]
        """
        if len(user_positions) == 0:
            return self._straight_line_strategy()
        
        # Get target position
        if self.environment_target is not None:
            target_position = self.environment_target
        else:
            target_position = self.target_position
        
        # Calculate center between users and target (weighted toward target)
        user_center = np.mean(user_positions, axis=0)
        # Weight: 30% users, 70% target progression
        effective_center = 0.3 * user_center + 0.7 * target_position
        
        # Use larger radius for better coverage
        radius = max(30.0, np.linalg.norm(user_center - target_position) * 0.2)
        
        # Calculate current position relative to effective center
        relative_pos = self.current_position - effective_center
        current_distance = np.linalg.norm(relative_pos[:2])  # Only X-Y for 2D circle
        
        if current_distance < 5.0:
            # Too close to center, move outward
            angle = np.random.uniform(0, 2*np.pi)
            target_pos = effective_center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0
            ])
        else:
            # Move in arc toward target
            current_angle = np.arctan2(relative_pos[1], relative_pos[0])
            
            # Calculate direction toward target
            target_direction = target_position - effective_center
            target_angle = np.arctan2(target_direction[1], target_direction[0])
            
            # Move in arc toward target direction
            angle_diff = target_angle - current_angle
            # Normalize angle difference to [-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2*np.pi
            while angle_diff < -np.pi:
                angle_diff += 2*np.pi
            
            # Small step toward target angle
            next_angle = current_angle + np.sign(angle_diff) * min(abs(angle_diff), 0.15)
            
            target_pos = effective_center + np.array([
                radius * np.cos(next_angle),
                radius * np.sin(next_angle),
                0.0
            ])
        
        # Move towards target position
        direction = target_pos - self.current_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            normalized_direction = direction / distance
            speed = min(25.0, max(10.0, distance * 0.4))  # Within valid range [10, 30]
            
            # Hover if close to any user
            min_user_distance = min([np.linalg.norm(self.current_position - user_pos) for user_pos in user_positions])
            hover = 0.7 if min_user_distance < 35.0 else 0.0
        else:
            normalized_direction = np.array([0.0, 0.0, 0.0])
            speed = 10.0  # Minimum valid speed
            hover = 0.0
        
        return np.array([normalized_direction[0], normalized_direction[1], speed, hover])
    
    def _random_strategy(self) -> np.ndarray:
        """
        Random strategy: move in random direction.
        
        Returns:
            Action array [dx, dy, speed, hover]
        """
        # Random direction
        direction = np.random.uniform(-1, 1, 2)
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1, 0])
        
        # Random speed within valid range
        speed = np.random.uniform(10.0, 30.0)  # Valid range [10, 30]
        
        # Random hover decision
        hover = np.random.choice([0.0, 1.0], p=[0.8, 0.2])  # 20% chance to hover
        
        return np.array([direction[0], direction[1], speed, hover])
    
    def _get_num_users(self) -> int:
        """Get number of users from observation space."""
        # Observation: [uav_pos(3), remaining_time(1), user_positions(num_users*3), throughput_history(5)]
        obs_dim = self.observation_space.shape[0]
        user_pos_dim = obs_dim - 3 - 1 - 5  # Total - uav_pos - remaining_time - throughput_history
        return user_pos_dim // 3
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update method for baseline agent (no learning).
        
        Args:
            batch: Training batch (not used)
            
        Returns:
            Empty dictionary (no learning)
        """
        return {}
    
    def reset(self) -> None:
        """Reset the baseline agent."""
        self.current_position = self.start_position.copy()
        self.target_position = self.end_position.copy()
        self.circular_center = None
    
    def set_target(self, target_position: np.ndarray) -> None:
        """
        Set target position for the agent.
        
        Args:
            target_position: New target position
        """
        self.target_position = np.array(target_position)
    
    def set_start(self, start_position: np.ndarray) -> None:
        """
        Set start position for the agent.
        
        Args:
            start_position: New start position
        """
        self.start_position = np.array(start_position)
        self.current_position = self.start_position.copy()
    
    def get_action_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Get additional information about action selection.
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary containing action information
        """
        return {
            'strategy': self.strategy,
            'current_position': self.current_position.copy(),
            'target_position': self.target_position.copy(),
            'distance_to_target': np.linalg.norm(self.target_position - self.current_position)
        }
    
    def __repr__(self) -> str:
        return f"BaselineAgent(strategy={self.strategy})"


class StraightLineAgent(BaselineAgent):
    """Straight line baseline agent."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, strategy='straight_line', **kwargs)


class GreedyAgent(BaselineAgent):
    """Greedy baseline agent."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, strategy='greedy', **kwargs)


class CircularAgent(BaselineAgent):
    """Circular baseline agent."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, strategy='circular', **kwargs)


class RandomAgent(BaselineAgent):
    """Random baseline agent."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, strategy='random', **kwargs)


# Benchmark-specific agents for the four scenarios
class Benchmark1Agent(BaselineAgent):
    """Benchmark 1: Straight line trajectory + Optimized beamforming."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, strategy='straight_line', beamforming_type='optimized', **kwargs)

class Benchmark2Agent(BaselineAgent):
    """Benchmark 2: Straight line trajectory + Random beamforming."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, strategy='straight_line', **kwargs):
        super().__init__(observation_space, action_space, strategy='straight_line', beamforming_type='random', **kwargs)