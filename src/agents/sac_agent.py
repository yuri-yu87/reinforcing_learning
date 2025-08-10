"""
SAC (Soft Actor-Critic) agent implementation.
Uses stable-baselines3 for the core algorithm.
"""

import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .base_agent import BaseAgent


class SACCallback(BaseCallback):
    """Custom callback for SAC training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if self.locals.get('dones') is not None:
            # Log episode information
            for i, done in enumerate(self.locals['dones']):
                if done:
                    episode_reward = self.locals.get('rewards', [0])[i]
                    episode_length = self.locals.get('infos', [{}])[i].get('episode', {}).get('l', 0)
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)


class SACAgent(BaseAgent):
    """
    SAC agent implementation using stable-baselines3.
    
    SAC is an off-policy algorithm that maximizes both expected return
    and entropy, making it suitable for continuous control tasks like UAV navigation.
    """
    
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 target_update_interval: int = 1,
                 ent_coef: str = 'auto',
                 target_entropy: str = 'auto',
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 use_sde_at_warmup: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: str = 'auto',
                 **kwargs):
        """
        Initialize SAC agent.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            learning_rate: Learning rate
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            tau: Soft update coefficient
            gamma: Discount factor
            train_freq: Training frequency
            gradient_steps: Number of gradient steps per update
            target_update_interval: Target network update interval
            ent_coef: Entropy coefficient
            target_entropy: Target entropy
            use_sde: Whether to use State Dependent Exploration
            sde_sample_freq: Sample frequency for SDE
            use_sde_at_warmup: Whether to use SDE at warmup
            policy_kwargs: Policy network kwargs
            verbose: Verbosity level
            seed: Random seed
            device: Device to use
        """
        super().__init__(observation_space, action_space, **kwargs)
        
        # SAC specific parameters
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.use_sde_at_warmup = use_sde_at_warmup
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.seed = seed
        self.device = device
        
        # Initialize model and environment
        self.model = None
        self.vec_env = None
        self.callback = None
    
    def _create_model(self, env) -> None:
        """Create SAC model with vectorized environment."""
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Normalize observations
        self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        # Create SAC model
        self.model = SAC(
            policy='MlpPolicy',
            env=self.vec_env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            ent_coef=self.ent_coef,
            target_entropy=self.target_entropy,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            use_sde_at_warmup=self.use_sde_at_warmup,
            policy_kwargs=self.policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device
        )
        
        # Create callback
        self.callback = SACCallback(verbose=self.verbose)
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using SAC policy.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action
        """
        if self.model is None:
            # Random action if model not initialized
            return self.action_space.sample()
        
        # Normalize observation
        if self.vec_env is not None:
            observation = self.vec_env.normalize_obs(observation)
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update SAC model.
        
        SAC handles its own updates internally through the replay buffer,
        so this method is mainly for compatibility with the base agent interface.
        
        Args:
            batch: Batch of experience (not used for SAC)
            
        Returns:
            Dictionary of training metrics
        """
        # SAC handles updates internally
        return {}
    
    def setup_model(self, env) -> None:
        """Setup SAC model with environment."""
        self._create_model(env)
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """
        Train SAC agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for training
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Use provided callback or default
        training_callback = callback or self.callback
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=training_callback,
            log_interval=10,
            reset_num_timesteps=False
        )
        
        # Update training stats
        self.total_steps += total_timesteps
    
    def save(self, filepath: str) -> None:
        """Save SAC model."""
        if self.model is not None:
            self.model.save(filepath)
            # Also save normalization stats
            if self.vec_env is not None:
                self.vec_env.save(f"{filepath}_vecnormalize.pkl")
    
    def load(self, filepath: str) -> None:
        """Load SAC model."""
        if self.vec_env is not None:
            self.model = SAC.load(filepath, env=self.vec_env)
            # Try to load normalization stats
            try:
                self.vec_env = VecNormalize.load(f"{filepath}_vecnormalize.pkl", self.vec_env)
            except FileNotFoundError:
                print("Warning: Could not load normalization stats")
        else:
            print("Warning: Environment not set up. Call setup_model() first.")
    
    def get_action_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """Get detailed action information."""
        if self.model is None:
            return {'action': self.action_space.sample(), 'log_prob': 0.0}
        
        # Normalize observation
        if self.vec_env is not None:
            observation = self.vec_env.normalize_obs(observation)
        
        action, _ = self.model.predict(observation, deterministic=False)
        
        return {
            'action': action,
            'deterministic_action': self.model.predict(observation, deterministic=True)[0],
            'policy_type': 'SAC'
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = super().get_training_stats()
        
        if self.callback is not None:
            stats.update({
                'sac_episode_rewards': self.callback.episode_rewards,
                'sac_episode_lengths': self.callback.episode_lengths,
                'sac_avg_episode_reward': np.mean(self.callback.episode_rewards) if self.callback.episode_rewards else 0.0,
                'sac_avg_episode_length': np.mean(self.callback.episode_lengths) if self.callback.episode_lengths else 0.0
            })
        
        return stats