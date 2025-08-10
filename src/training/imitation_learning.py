#!/usr/bin/env python3
"""
Imitation Learning for UAV navigation.
Train RL agents to first imitate successful baseline trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import pickle
import os


class TrajectoryDataset(Dataset):
    """Dataset for storing expert trajectories."""
    
    def __init__(self, observations: List[np.ndarray], actions: List[np.ndarray]):
        """
        Initialize trajectory dataset.
        
        Args:
            observations: List of observation arrays
            actions: List of action arrays
        """
        self.observations = observations
        self.actions = actions
        
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = torch.FloatTensor(self.observations[idx])
        action = torch.FloatTensor(self.actions[idx])
        return obs, action


class ImitationPolicy(nn.Module):
    """Neural network policy for imitation learning."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize imitation policy network.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        self.action_dim = action_dim
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(obs)
    
    def get_action(self, obs: np.ndarray, action_space) -> np.ndarray:
        """Get action from observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            raw_action = self.forward(obs_tensor).squeeze(0).numpy()
        
        # Scale action to action space bounds
        low = action_space.low
        high = action_space.high
        scaled_action = low + (raw_action + 1) * (high - low) / 2
        
        return np.clip(scaled_action, low, high)


class ImitationLearner:
    """Imitation learning trainer."""
    
    def __init__(self, obs_dim: int, action_dim: int, device: str = 'cpu'):
        """
        Initialize imitation learner.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Training device
        """
        self.device = device
        self.policy = ImitationPolicy(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
    def train(self, dataset: TrajectoryDataset, epochs: int = 100, batch_size: int = 32) -> List[float]:
        """
        Train the imitation policy.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            List of training losses
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        
        self.policy.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for obs, actions in dataloader:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                predicted_actions = self.policy(obs)
                loss = self.criterion(predicted_actions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        return losses
    
    def save_policy(self, path: str):
        """Save the trained policy."""
        torch.save(self.policy.state_dict(), path)
        
    def load_policy(self, path: str):
        """Load a trained policy."""
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()


def collect_expert_trajectories(env, expert_agent, num_episodes: int = 20) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect expert trajectories from baseline agent.
    
    Args:
        env: Environment
        expert_agent: Expert baseline agent
        num_episodes: Number of episodes to collect
        
    Returns:
        Tuple of (observations, actions)
    """
    observations = []
    actions = []
    
    successful_episodes = 0
    episode_count = 0
    
    print(f"🎯 Collecting expert trajectories...")
    
    while successful_episodes < num_episodes and episode_count < num_episodes * 3:
        episode_count += 1
        obs, _ = env.reset(seed=42 + episode_count)
        
        episode_obs = []
        episode_actions = []
        
        # Set expert target
        expert_agent.set_target_position(env.end_position)
        
        max_steps = 80
        success = False
        
        for step in range(max_steps):
            # Get expert action
            action = expert_agent.select_action(obs)
            
            # Store observation and action
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                success = True
                print(f"   Episode {successful_episodes + 1}: SUCCESS in {step + 1} steps")
                break
                
            if truncated:
                break
        
        # Only keep successful trajectories
        if success:
            observations.extend(episode_obs)
            actions.extend(episode_actions)
            successful_episodes += 1
        else:
            # Check if got close
            final_pos = env.uav.get_position()
            final_distance = np.linalg.norm(final_pos - env.end_position)
            if final_distance < 15:  # Accept "close enough" trajectories
                observations.extend(episode_obs)
                actions.extend(episode_actions)
                successful_episodes += 1
                print(f"   Episode {successful_episodes}: PARTIAL SUCCESS ({final_distance:.1f}m)")
    
    print(f"✅ Collected {len(observations)} expert transitions from {successful_episodes} episodes")
    return observations, actions


def create_hybrid_agent(imitation_policy: ImitationPolicy, action_space, exploration_rate: float = 0.1):
    """
    Create a hybrid agent that uses imitation policy with some exploration.
    
    Args:
        imitation_policy: Trained imitation policy
        action_space: Environment action space
        exploration_rate: Rate of random exploration
        
    Returns:
        Function that selects actions
    """
    def select_action(obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and np.random.random() < exploration_rate:
            # Random exploration
            return action_space.sample()
        else:
            # Use imitation policy
            return imitation_policy.get_action(obs, action_space)
    
    return select_action

