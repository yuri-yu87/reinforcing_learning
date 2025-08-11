"""
Training configuration classes.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Environment parameters
    env_size: tuple = (100, 100, 50)
    num_users: int = 2
    num_antennas: int = 8
    episode_length: int = 200
    transmit_power: float = 0.5
    
    # Agent parameters
    agent_type: str = 'dqn'  # 'ppo', 'baseline'
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    
    # Training parameters
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    log_freq: int = 1000
    
    # Joint optimization parameters
    use_joint_optimization: bool = False
    beamforming_method: str = 'mrt'
    power_optimization_strategy: str = 'efficiency'
    
    # Legacy benchmark parameters (for backward compatibility)
    benchmark_scenario: Optional[str] = None  # Deprecated: use agent_type instead
    
    # Baseline parameters
    baseline_strategy: str = 'straight_line'  # 'straight_line', 'greedy', 'circular', 'random'
    
    # Logging parameters
    log_dir: str = './logs'
    tensorboard_log: Optional[str] = None
    verbose: int = 1
    
    # Evaluation parameters
    eval_episodes: int = 10
    eval_deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'env_size': self.env_size,
            'num_users': self.num_users,
            'num_antennas': self.num_antennas,
            'episode_length': self.episode_length,
            'transmit_power': self.transmit_power,
            'agent_type': self.agent_type,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'n_steps': self.n_steps,
            'n_epochs': self.n_epochs,
            'total_timesteps': self.total_timesteps,
            'eval_freq': self.eval_freq,
            'save_freq': self.save_freq,
            'log_freq': self.log_freq,
            'use_joint_optimization': self.use_joint_optimization,
            'beamforming_method': self.beamforming_method,
            'power_optimization_strategy': self.power_optimization_strategy,
            'benchmark_scenario': self.benchmark_scenario,
            'baseline_strategy': self.baseline_strategy,
            'log_dir': self.log_dir,
            'tensorboard_log': self.tensorboard_log,
            'verbose': self.verbose,
            'eval_episodes': self.eval_episodes,
            'eval_deterministic': self.eval_deterministic
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class PPOConfig:
    """PPO-specific configuration."""
    
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    tensorboard_log: Optional[str] = None
    policy_kwargs: Optional[Dict[str, Any]] = None
    verbose: int = 0


@dataclass
class BaselineConfig:
    """Baseline-specific configuration."""
    
    strategy: str = 'straight_line'  # 'straight_line', 'greedy', 'circular', 'random'
    circular_radius: float = 20.0


# Predefined configurations
def get_ppo_config() -> TrainingConfig:
    """Get PPO training configuration."""
    return TrainingConfig(
        agent_type='ppo',
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        total_timesteps=1000000,
        eval_freq=10000,
        save_freq=50000,
        log_freq=1000,
        use_joint_optimization=False,
        beamforming_method='mrt',
        power_optimization_strategy='efficiency'
    )


def get_baseline_config() -> TrainingConfig:
    """Get baseline training configuration."""
    return TrainingConfig(
        agent_type='baseline',
        baseline_strategy='straight_line',
        total_timesteps=10000,  # Shorter for baseline
        eval_freq=1000,
        save_freq=5000,
        log_freq=100
    )


def get_joint_optimization_config() -> TrainingConfig:
    """Get joint optimization training configuration."""
    return TrainingConfig(
        agent_type='ppo',
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        total_timesteps=1000000,
        eval_freq=10000,
        save_freq=50000,
        log_freq=1000,
        use_joint_optimization=True,
        beamforming_method='mrt',
        power_optimization_strategy='efficiency'
    )


def get_quick_test_config() -> TrainingConfig:
    """Get quick test configuration for debugging."""
    return TrainingConfig(
        agent_type='ppo',
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=32,
        n_steps=1024,
        n_epochs=5,
        total_timesteps=100000,  # Short training
        eval_freq=5000,
        save_freq=25000,
        log_freq=500,
        use_joint_optimization=False,
        beamforming_method='mrt',
        power_optimization_strategy='efficiency'
    ) 