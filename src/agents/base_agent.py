"""
Base agent class for RL algorithms.
Defines the common interface for all agents.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym


class BaseAgent(ABC):
    """
    Base class for all RL agents.
    
    This class defines the common interface that all agents must implement.
    """
    
    def __init__(self, 
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 **kwargs):
        """
        Initialize the base agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            **kwargs: Additional arguments
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.training = True
        self.device = 'cpu'  # Default device
        
        # Training statistics
        self.total_steps = 0
        self.episode_count = 0
        self.training_rewards = []
        self.evaluation_rewards = []
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update the agent's policy/value networks.
        
        Args:
            batch: Training batch containing observations, actions, rewards, etc.
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save the agent's model to a file.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load the agent's model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def train(self) -> None:
        """Set the agent to training mode."""
        self.training = True
    
    def eval(self) -> None:
        """Set the agent to evaluation mode."""
        self.training = False
    
    def reset(self) -> None:
        """Reset the agent's internal state."""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'avg_training_reward': np.mean(self.training_rewards) if self.training_rewards else 0.0,
            'avg_evaluation_reward': np.mean(self.evaluation_rewards) if self.evaluation_rewards else 0.0
        }
    
    def log_episode_reward(self, reward: float, is_evaluation: bool = False) -> None:
        """
        Log episode reward for statistics.
        
        Args:
            reward: Episode reward
            is_evaluation: Whether this is an evaluation episode
        """
        if is_evaluation:
            self.evaluation_rewards.append(reward)
        else:
            self.training_rewards.append(reward)
            self.episode_count += 1
    
    def get_action_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Get additional information about action selection.
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary containing action information
        """
        return {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(obs_space={self.observation_space}, action_space={self.action_space})" 