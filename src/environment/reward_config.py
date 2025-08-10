"""
Reward Configuration for UAV Environment

This module defines reward parameters and configuration for the UAV environment.
Following the architectural principle: reward mechanism belongs to Environment layer.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RewardConfig:
    """
    Reward system configuration for UAV environment.
    
    Implements the optimized reward formula:
    r = w_rate · normalize(sum_rate) + w_goal · [F(d_end(s')) - F(d_end(s))] 
        + w_fair · Σ log(ε + service_i) - w_time · Δt
    """
    
    # Main reward weights
    w_rate: float = 1.0                    # Throughput weight (main objective)
    w_goal: float = 0.3                    # Mission progress weight  
    w_fair: float = 0.2                    # Fairness weight (encourage user rotation)
    w_time: float = 0.02                   # Time penalty weight (gentle)
    
    # Normalization parameters
    max_expected_throughput: float = 8.0   # For throughput normalization
    distance_normalization: float = 50.0   # Distance normalization parameter (d0)
    fairness_epsilon: float = 1e-6        # Small value to avoid log(0)
    
    # Mission completion tolerance
    end_position_tolerance: float = 5.0    # Meters tolerance for mission completion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'w_rate': self.w_rate,
            'w_goal': self.w_goal, 
            'w_fair': self.w_fair,
            'w_time': self.w_time,
            'max_expected_throughput': self.max_expected_throughput,
            'distance_normalization': self.distance_normalization,
            'fairness_epsilon': self.fairness_epsilon,
            'end_position_tolerance': self.end_position_tolerance
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def get_high_throughput_config(cls) -> 'RewardConfig':
        """Configuration optimized for high throughput."""
        return cls(
            w_rate=1.5,      # Higher throughput weight
            w_goal=0.2,      # Lower goal weight
            w_fair=0.1,      # Lower fairness weight
            w_time=0.01
        )
    
    @classmethod
    def get_balanced_config(cls) -> 'RewardConfig':
        """Balanced configuration for throughput + fairness."""
        return cls(
            w_rate=1.0,
            w_goal=0.3,
            w_fair=0.3,      # Higher fairness weight
            w_time=0.02
        )
    
    @classmethod
    def get_exploration_config(cls) -> 'RewardConfig':
        """Configuration for exploration and user coverage."""
        return cls(
            w_rate=0.8,
            w_goal=0.4,      # Higher goal weight
            w_fair=0.4,      # Higher fairness weight  
            w_time=0.05      # Higher time pressure
        )


class RewardCalculator:
    """
    Reward calculation logic for UAV environment.
    
    This class implements the actual reward computation based on the configuration.
    It maintains internal state for incremental potential function shaping.
    """
    
    def __init__(self, config: RewardConfig):
        """Initialize reward calculator with configuration."""
        self.config = config
        self.previous_distance_to_end = None
        self.user_cumulative_service = {}
    
    def reset(self, num_users: int):
        """Reset internal state for new episode."""
        self.previous_distance_to_end = None
        self.user_cumulative_service = {i: 0.0 for i in range(num_users)}
    
    def potential_function(self, distance_to_end: float) -> float:
        """Potential function: F(d) = 1/(1 + d/d0)"""
        return 1.0 / (1.0 + distance_to_end / self.config.distance_normalization)
    
    def calculate_reward(self, 
                        current_throughput: float,
                        uav_position,
                        end_position,
                        user_individual_throughputs,
                        time_step: float) -> tuple[float, Dict[str, Any]]:
        """
        Calculate reward based on current state.
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        import numpy as np
        
        # 1. Normalized throughput reward (main objective)
        normalized_throughput = np.clip(
            current_throughput / self.config.max_expected_throughput, 0.0, 1.0
        )
        throughput_reward = self.config.w_rate * normalized_throughput
        
        # 2. Mission progress reward (potential function increment)
        current_distance = np.linalg.norm(uav_position - end_position)
        
        if self.previous_distance_to_end is not None:
            # Potential function shaping: γF(s') - F(s)
            F_current = self.potential_function(current_distance)
            F_previous = self.potential_function(self.previous_distance_to_end)
            goal_reward = self.config.w_goal * (F_current - F_previous)
        else:
            goal_reward = 0.0  # No increment for first step
        
        self.previous_distance_to_end = current_distance
        
        # 3. Fairness reward (encourage all users to receive service)
        fair_reward = 0.0
        if len(user_individual_throughputs) > 0:
            # Update cumulative service quality
            for i, rate in enumerate(user_individual_throughputs):
                if i in self.user_cumulative_service:
                    self.user_cumulative_service[i] += rate * time_step
            
            # Fairness utility: Σ log(ε + service_i)
            fair_utilities = [
                np.log(self.config.fairness_epsilon + service) 
                for service in self.user_cumulative_service.values()
            ]
            fair_reward = self.config.w_fair * np.sum(fair_utilities)
        
        # 4. Time penalty (gentle)
        time_penalty = -self.config.w_time * time_step
        
        # Total reward (no clipping for more natural learning)
        total_reward = throughput_reward + goal_reward + fair_reward + time_penalty
        
        # Detailed breakdown for debugging/analysis
        reward_breakdown = {
            'throughput_reward': float(throughput_reward),
            'goal_reward': float(goal_reward),
            'fair_reward': float(fair_reward),
            'time_penalty': float(time_penalty),
            'total_reward': float(total_reward),
            'normalized_throughput': float(normalized_throughput),
            'distance_to_end': float(current_distance),
            'cumulative_services': dict(self.user_cumulative_service)
        }
        
        return float(total_reward), reward_breakdown
