"""
PPO (Proximal Policy Optimization) agent implementation.
Uses stable-baselines3 for the core algorithm.
"""

import numpy as np
from typing import Dict, Any, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .base_agent import BaseAgent


class PPOCallback(BaseCallback):
    """Custom callback for PPO training."""
    
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


class PPOAgent(BaseAgent):
    """
    PPO agent implementation using stable-baselines3.
    
    PPO is a policy gradient method that is stable and easy to tune,
    making it suitable for UAV trajectory optimization.
    """
    
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 learning_rate: float = 5e-4,  # Increased learning rate
                 n_steps: int = 4096,  # Increased buffer size
                 batch_size: int = 128,  # Increased batch size
                 n_epochs: int = 15,  # More epochs per update
                 gamma: float = 0.995,  # Higher discount factor for long episodes
                 gae_lambda: float = 0.98,  # Higher GAE lambda
                 clip_range: float = 0.15,  # Smaller clip range for stability
                 clip_range_vf: Optional[float] = None,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.01,  # Small entropy coefficient to encourage exploration
                 vf_coef: float = 0.8,  # Higher value function coefficient
                 max_grad_norm: float = 0.3,  # Smaller gradient clipping
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = 0.02,  # Target KL divergence
                 tensorboard_log: Optional[str] = None,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 beamforming_type: str = 'optimized',  # 'optimized' or 'random'
                 **kwargs):
        """
        Initialize PPO agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            learning_rate: Learning rate for the optimizer
            n_steps: Number of steps to run for each environment per update
            batch_size: Minibatch size for each update
            n_epochs: Number of epochs when optimizing the surrogate loss
            gamma: Discount factor
            gae_lambda: Factor for trade-off of bias vs variance for GAE
            clip_range: Clipping parameter for the value function
            clip_range_vf: Clipping parameter for the value function
            normalize_advantage: Whether to normalize the advantage
            ent_coef: Entropy coefficient for the loss calculation
            vf_coef: Value function coefficient for the loss calculation
            max_grad_norm: Maximum value for the gradient clipping
            use_sde: Whether to use generalized State Dependent Exploration
            sde_sample_freq: Sample a new noise matrix every n steps
            target_kl: Target KL divergence between old and new policy
            tensorboard_log: Log directory for tensorboard
            policy_kwargs: Additional arguments for the policy
            verbose: Verbosity level
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space, **kwargs)
        
        # Store PPO parameters
        self.learning_rate = learning_rate
        self.beamforming_type = beamforming_type
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        
        # Initialize PPO model
        self.model = None
        self.vec_env = None
        self.callback = None
        
    def _create_model(self, env) -> None:
        """Create the PPO model."""
        # Create vectorized environment
        if not hasattr(env, 'num_envs'):
            env = DummyVecEnv([lambda: env])
        
        # Add normalization if needed
        if self.normalize_advantage:
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create PPO model with device detection
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 PPO Agent using device: {device}")
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            normalize_advantage=self.normalize_advantage,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            target_kl=self.target_kl,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=self.policy_kwargs,
            verbose=self.verbose,
            device=device
        )
        
        self.vec_env = env
        self.callback = PPOCallback(verbose=self.verbose)
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """
        Select an action using the PPO policy.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action (scalar for discrete action space, array for continuous)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Ensure observation is in the right format
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
        
        # Get action from model
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Update step count
        self.total_steps += 1
        
        # **FIX: Handle discrete vs continuous action spaces properly**
        from gymnasium import spaces
        if isinstance(self.action_space, spaces.Discrete):
            # For discrete action space, return scalar integer
            return int(action.item())
        else:
            # For continuous action space, return flattened array
            return action.flatten()
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update the PPO model.
        
        Note: PPO uses its own training loop, so this method
        is mainly for compatibility with the base agent interface.
        
        Args:
            batch: Training batch (not used in PPO)
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            return {}
        
        # PPO handles its own updates internally
        # This method is mainly for compatibility
        return {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0
        }
    
    def setup_model(self, env) -> None:
        """
        Setup the PPO model with the environment.
        
        Args:
            env: Training environment
        """
        self._create_model(env)
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for training
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        # Combine callbacks
        callbacks = [self.callback]
        if callback is not None:
            callbacks.append(callback)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    
    def save(self, filepath: str) -> None:
        """
        Save the PPO model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load the PPO model.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = PPO.load(filepath)
    
    def get_action_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Get additional information about action selection.
        
        Args:
            observation: Current observation
            
        Returns:
            Dictionary containing action information
        """
        if self.model is None:
            return {}
        
        # Get action and value from model
        action, states = self.model.predict(observation, deterministic=False)
        
        return {
            'action': action,
            'states': states,
            'model_info': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'clip_range': self.clip_range
            },
            'beamforming_type': getattr(self, 'beamforming_type', 'optimized')
        }
    
    def set_beamforming_type(self, beamforming_type: str):
        """Set beamforming type for the agent."""
        if beamforming_type not in ['optimized', 'random']:
            raise ValueError(f"Unknown beamforming type: {beamforming_type}")
        self.beamforming_type = beamforming_type
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        stats = super().get_training_stats()
        
        # Add PPO-specific stats
        if self.callback is not None:
            stats.update({
                'avg_episode_length': np.mean(self.callback.episode_lengths) if self.callback.episode_lengths else 0.0,
                'total_episodes': len(self.callback.episode_rewards)
            })
        
        return stats
    
    def __repr__(self) -> str:
        return (f"PPOAgent(learning_rate={self.learning_rate}, "
                f"n_steps={self.n_steps}, batch_size={self.batch_size}, "
                f"gamma={self.gamma}, clip_range={self.clip_range})")


# Benchmark-specific PPO agents for the four scenarios
class Benchmark3Agent(PPOAgent):
    """Benchmark 3: RL trajectory + Random beamforming."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, beamforming_type='random', **kwargs)

class Benchmark4Agent(PPOAgent):
    """Benchmark 4: RL trajectory + Optimized beamforming."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, **kwargs):
        super().__init__(observation_space, action_space, beamforming_type='optimized', **kwargs) 