"""
Simplified trainer for RL agents with integrated callbacks.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from ..environment.uav_env import UAVEnvironment
from ..agents import PPOAgent, BaselineAgent, Benchmark1Agent, Benchmark2Agent
from .configs import TrainingConfig


class TrainingCallback(BaseCallback):
    """Integrated callback for training monitoring."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_start_time = time.time()
        
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    episode_reward = self.locals.get('rewards', [0])[i]
                    episode_length = self.locals.get('infos', [{}])[i].get('episode', {}).get('l', 0)
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        training_time = time.time() - self.training_start_time
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'training_time': training_time,
            'episodes_per_second': len(self.episode_rewards) / training_time if training_time > 0 else 0.0
        }


class Trainer:
    """
    Simplified trainer for RL agents.
    
    This class handles basic training and evaluation for different types of agents.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.env = None
        self.agent = None
        self.callback = None
        self.training_stats = {}
        
    def setup_environment(self) -> UAVEnvironment:
        """
        Setup the training environment.
        
        Returns:
            Configured environment
        """
        # Determine beamforming configuration based on agent type
        use_optimized_beamforming = True
        beamforming_method = self.config.beamforming_method
        
        # Handle legacy benchmark_scenario if present
        if hasattr(self.config, 'benchmark_scenario'):
            if 'random_beam' in self.config.benchmark_scenario:
                use_optimized_beamforming = False
                beamforming_method = 'random'
        
        self.env = UAVEnvironment(
            env_size=self.config.env_size,
            num_users=self.config.num_users,
            num_antennas=self.config.num_antennas,
            start_position=(10, 10, 50),
            end_position=(80, 80, 50),
            flight_time=250.0,
            time_step=0.1,
            transmit_power=self.config.transmit_power,
            max_speed=30.0,
            min_speed=10.0,
            beamforming_method=beamforming_method,
            power_optimization_strategy=self.config.power_optimization_strategy,
            use_optimized_beamforming=use_optimized_beamforming,
            fixed_users=True,
            seed=42
        )
        
        return self.env
    
    def setup_agent(self) -> None:
        """Setup the agent based on configuration."""
        if self.env is None:
            self.setup_environment()
        
        if self.config.agent_type == 'ppo':
            self.agent = PPOAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                tensorboard_log=self.config.tensorboard_log,
                verbose=self.config.verbose
            )
            # Setup PPO model
            self.agent.setup_model(self.env)
            
        elif self.config.agent_type == 'baseline':
            self.agent = BaselineAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                strategy=self.config.baseline_strategy
            )
        elif self.config.agent_type == 'benchmark_1':
            self.agent = Benchmark1Agent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
        elif self.config.agent_type == 'benchmark_2':
            self.agent = Benchmark2Agent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
            
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the agent.
        
        Returns:
            Training results
        """
        if self.agent is None:
            self.setup_agent()
        
        print(f"Starting training with {self.config.agent_type} agent...")
        
        start_time = time.time()
        
        if self.config.agent_type == 'ppo':
            # PPO has its own training loop
            self.callback = TrainingCallback(verbose=self.config.verbose)
            self.agent.train(
                total_timesteps=self.config.total_timesteps,
                callback=self.callback
            )
            
        elif self.config.agent_type == 'baseline':
            # Baseline agents don't need training, just evaluation
            self._evaluate_baseline()
        
        training_time = time.time() - start_time
        
        # Collect training statistics
        self.training_stats = {
            'training_time': training_time,
            'total_timesteps': self.config.total_timesteps,
            'agent_type': self.config.agent_type
        }
        
        if self.callback is not None:
            self.training_stats.update(self.callback.get_stats())
        
        print(f"Training completed in {training_time:.2f} seconds")
        return self.training_stats
    
    def _evaluate_baseline(self) -> None:
        """Evaluate baseline agent (no training needed)."""
        print("Evaluating baseline agent...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.config.eval_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: reward={episode_reward:.2f}, length={episode_length}")
        
        # Store evaluation results
        self.training_stats.update({
            'avg_episode_reward': np.mean(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        })
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Evaluation results
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
        
        print(f"Evaluating agent over {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_throughputs = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_throughput = 0
            
            while True:
                action = self.agent.select_action(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_throughput += info.get('average_throughput', 0)
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_throughputs.append(episode_throughput)
            
            if episode % 5 == 0:
                print(f"Episode {episode}: reward={episode_reward:.2f}, "
                      f"length={episode_length}, throughput={episode_throughput:.2f}")
        
        evaluation_results = {
            'avg_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'avg_episode_throughput': np.mean(episode_throughputs),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_throughputs': episode_throughputs,
            'num_episodes': num_episodes,
            'deterministic': deterministic
        }
        
        print(f"Evaluation completed:")
        print(f"  Average reward: {evaluation_results['avg_episode_reward']:.2f} ± {evaluation_results['std_episode_reward']:.2f}")
        print(f"  Average length: {evaluation_results['avg_episode_length']:.1f}")
        print(f"  Average throughput: {evaluation_results['avg_episode_throughput']:.2f}")
        
        return evaluation_results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.agent is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.agent.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        if self.agent is not None:
            self.agent.load(filepath)
            print(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return self.training_stats.copy()
    
    def __repr__(self) -> str:
        return f"Trainer(agent_type={self.config.agent_type}, timesteps={self.config.total_timesteps})" 