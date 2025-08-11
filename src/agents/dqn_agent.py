#!/usr/bin/env python3
"""
DQN Agent implementation specifically designed for discrete action spaces.

This DQN agent is optimized for discrete action environments like the 5-direction UAV control.
DQN is naturally suited for discrete actions, making it potentially more effective than PPO 
for this specific use case.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from collections import deque
import random
from gymnasium import spaces

from .base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action selection."""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: list = [256, 256]):
        """
        Initialize DQN network.
        
        Args:
            state_size: Dimension of observation space
            action_size: Number of discrete actions
            hidden_layers: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        # Output layer (Q-values for each action)
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for discrete action spaces.
    
    This agent is specifically designed for discrete action environments and should
    perform better than PPO for such tasks.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        hidden_layers: list = [256, 256],
        beamforming_type: str = 'mrt',
        **kwargs
    ):
        """
        Initialize DQN agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space (must be Discrete)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay epsilon
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update: Steps between target network updates
            hidden_layers: Hidden layer sizes
            beamforming_type: Type of beamforming ('mrt', 'random')
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space, **kwargs)
        
        # Validate action space
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("DQN requires discrete action space")
        
        # Agent parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.beamforming_type = beamforming_type
        
        # Network parameters
        self.state_size = observation_space.shape[0]
        self.action_size = action_space.n
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 DQN Agent using device: {self.device}")
        
        # Initialize networks
        self.q_network = DQNNetwork(self.state_size, self.action_size, hidden_layers).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size, hidden_layers).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training state
        self.epsilon = epsilon_start
        self.steps_done = 0
        self.training_step = 0
        
        # Training metrics
        self.losses = []
        self.q_values = []
        self.episode_rewards = []
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected discrete action (0-4)
        """
        # Update exploration rate
        if not deterministic:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.steps_done / self.epsilon_decay) * (self.epsilon_start - self.epsilon_end)
            )
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # Use Q-network to select action
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = q_values.max(1)[1].item()
        
        self.steps_done += 1
        self.total_steps += 1
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self, batch: Dict[str, np.ndarray] = None) -> Dict[str, float]:
        """
        Update the DQN networks.
        
        Args:
            batch: Training batch (optional, will sample from replay buffer if None)
            
        Returns:
            Dictionary containing training metrics
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Store metrics
        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'steps': self.steps_done
        }
    
    def train(self, total_timesteps: int, env, callback: Optional = None) -> None:
        """
        Train the DQN agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            env: Training environment
            callback: Optional callback for training
        """
        print(f"🚀 Starting DQN training for {total_timesteps} timesteps...")
        
        episode = 0
        step = 0
        
        while step < total_timesteps:
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and step < total_timesteps:
                # Select action
                action = self.select_action(obs)
                
                # Take step
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.store_experience(obs, action, reward, next_obs, done)
                
                # Update networks
                if step % 4 == 0:  # Update every 4 steps
                    metrics = self.update()
                
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                step += 1
                
                # Progress tracking
                if step % 1000 == 0:
                    print(f"Step {step}/{total_timesteps}, Episode {episode}, "
                          f"Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            episode += 1
            # Store episode reward for convergence curve plotting
            self.episode_rewards.append(float(episode_reward))
        
        print(f"✅ DQN training completed! Total episodes: {episode}")
    
    def save(self, filepath: str) -> None:
        """Save the DQN model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_step': self.training_step
        }, filepath)
        print(f"💾 DQN model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the DQN model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.training_step = checkpoint.get('training_step', 0)
        print(f"📂 DQN model loaded from {filepath}")
