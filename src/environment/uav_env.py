import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .uav import UAV
from .users import UserManager
from utils.channel import ChannelModel
from utils.signal import SignalProcessor


class UAVEnvironment(gym.Env):
    """
    Simplified UAV-aided telecommunication environment for reinforcement learning.
    
    This environment focuses on the core physics simulation and environment dynamics,
    following the principle of keeping environments simple and focused.
    
    Responsibilities (Environment Layer):
    - Physical state management (UAV position, time, boundaries)
    - Basic observation computation (positions, signal quality, time)
    - Simple reward calculation (throughput, mission progress, safety)
    - Termination condition checking
    - Component coordination (UAV + Users + Channel + Signal)
    
    Removed from Environment (moved to appropriate layers):
    - Complex reward strategies → Trainer layer
    - Business decision logic → Agent layer  
    - Performance analysis → Analysis module
    - Algorithm selection → Utils layer
    """
    
    def __init__(self,
                 env_size: Tuple[float, float, float] = (100, 100, 50),
                 num_users: int = 2,
                 num_antennas: int = 8,
                 start_position: Tuple[float, float, float] = (0, 0, 50),
                 end_position: Tuple[float, float, float] = (80, 80, 50),
                 flight_time: float = 250.0,
                 time_step: float = 0.1,
                 transmit_power: float = 0.5,
                 max_speed: float = 30.0,
                 min_speed: float = 10.0,
                 path_loss_exponent: float = 2.5,
                 noise_power: float = -100.0,
                 fixed_users: bool = True,
                 reward_config = None,
                 seed: Optional[int] = None):
        """
        Initialize simplified UAV environment.
        
        Args:
            env_size: Environment size (x, y, z)
            num_users: Number of ground users
            num_antennas: Number of UAV antennas
            start_position: UAV start position
            end_position: UAV end position
            flight_time: Total flight time in seconds
            time_step: Time step in seconds
            transmit_power: Total transmit power constraint
            max_speed: Maximum UAV speed
            min_speed: Minimum UAV speed
            path_loss_exponent: Path loss exponent
            noise_power: Noise power in dBm
            fixed_users: Use fixed user positions
            seed: Random seed
        """
        super().__init__()
        
        # Environment parameters
        self.env_size = env_size
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        
        # Time management
        self.flight_time = flight_time
        self.time_step = max(0.01, min(1.0, time_step))
        self.total_steps = int(flight_time / self.time_step)
        
        # Physical parameters
        self.transmit_power = transmit_power
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.path_loss_exponent = path_loss_exponent
        self.noise_power = noise_power
        
        # User position management
        self.fixed_users = fixed_users
        if fixed_users:
            # Fixed user positions for training
            base_positions = np.array([
                [15.0, 75.0, 0.0],   
                [75.0, 15.0, 0.0]    
            ])
            
            # Add small random variation for better generalization
            if seed is not None:
                np.random.seed(seed)
            random_offset = np.random.uniform(-5.0, 5.0, (2, 2))
            self.fixed_user_positions = base_positions.copy()
            self.fixed_user_positions[:, :2] += random_offset
        else:
            self.fixed_user_positions = None
        
        # Reward system configuration (Environment layer responsibility)
        from environment.reward_config import RewardConfig, RewardCalculator
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.reward_calculator = RewardCalculator(self.reward_config)
        
        # Initialize components
        self.uav = UAV(
            start_position=self.start_position,
            max_speed=max_speed,
            min_speed=min_speed,
            env_bounds=(self.env_size[0], self.env_size[1])
        )
        
        self.user_manager = UserManager(num_users=num_users)
        
        self.channel_model = ChannelModel(
            path_loss_exponent=path_loss_exponent,
            noise_power=noise_power,
            seed=seed
        )
        
        self.signal_processor = SignalProcessor(num_antennas=num_antennas)
        
        # Episode tracking (minimal)
        self.current_step = 0
        self.current_time = 0.0
        self.total_throughput = 0.0
        self.episode_throughput_history = []

        # Default transmission strategy (can be changed via setter)
        self.default_beamforming_method: str = 'mrt'
        self.default_power_strategy: str = 'proportional'
        

        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
        
        # Define action and observation spaces
        self._setup_spaces()
    
    def set_transmit_strategy(self,
                              beamforming_method: Optional[str] = None,
                              power_strategy: Optional[str] = None) -> None:
        """
        Configure default beamforming and power allocation strategies used in throughput calculation.
        If not set, environment defaults are used ('mrt' + 'proportional').
        """
        if beamforming_method is not None:
            self.default_beamforming_method = beamforming_method
        if power_strategy is not None:
            self.default_power_strategy = power_strategy

    def _setup_spaces(self):
        """Setup action and observation spaces for the environment."""
        
        # Discrete action space with 5 basic directions
        # Action mapping: 0=East(+X), 1=South(-Y), 2=West(-X), 3=North(+Y), 4=Hover
        self.action_space = spaces.Discrete(5)
        
        # Define action mappings for discrete actions
        self.discrete_actions = {
            0: np.array([1.0, 0.0, self.max_speed]),   # East: +X direction
            1: np.array([0.0, -1.0, self.max_speed]),  # South: -Y direction  
            2: np.array([-1.0, 0.0, self.max_speed]),  # West: -X direction
            3: np.array([0.0, 1.0, self.max_speed]),   # North: +Y direction
            4: np.array([0.0, 0.0, 0.0])               # Hover: no movement
        }
        
        # Observation space: simplified and focused
        # [uav_x, uav_y, uav_z, end_x, end_y, end_z, remaining_time_ratio, 
        #  signal_quality_user_0, signal_quality_user_1, ..., current_throughput]
        obs_dim = 3 + 3 + 1 + self.num_users + 1  # UAV(3) + End(3) + Time(1) + Users(N) + Throughput(1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple[np.ndarray, Dict]: Initial observation and info
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset UAV to start position
        self.uav.reset(self.start_position)
        
        # Set user positions
        if self.fixed_users:
            self.user_manager.set_user_positions(self.fixed_user_positions)
        else:
            self.user_manager.generate_random_users(
                x_range=(0, self.env_size[0]),
                y_range=(0, self.env_size[1]),
                seed=seed
            )
        
        # Reset episode tracking
        self.current_step = 0
        self.current_time = 0.0
        self.total_throughput = 0.0
        
        # Reset reward calculator state
        self.reward_calculator.reset(self.num_users)
        self.episode_throughput_history = []
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action integer (0=East, 1=South, 2=West, 3=North, 4=Hover)
            
        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert discrete action to continuous action format
        if action in self.discrete_actions:
            continuous_action = self.discrete_actions[action]
        else:
            # Default to hover if invalid action
            continuous_action = self.discrete_actions[4]
        
        # Parse action
        movement_direction = continuous_action[:2]
        speed = continuous_action[2]
        
        # Move UAV
        if speed > 0:
            self.uav.move(movement_direction, speed, self.time_step)
        # If speed is 0, UAV hovers (no movement)
        
        # Calculate throughput for current position
        current_throughput = self._calculate_throughput(
            beamforming_method=getattr(self, 'default_beamforming_method', 'mrt'),
            power_strategy=getattr(self, 'default_power_strategy', 'proportional')
        )
        self.total_throughput += current_throughput
        self.episode_throughput_history.append(current_throughput)
        
        # Update time and step counter
        self.current_step += 1
        self.current_time += self.time_step
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_time >= self.flight_time
        
        # Calculate simple, focused reward
        reward = self._calculate_reward(current_throughput)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_throughput(self,
                              beamforming_method: Optional[str] = None,
                              power_strategy: Optional[str] = None) -> float:
        """
        Calculate total throughput for current UAV position.
        
        Simplified: Use signal processor's unified interface instead of 
        selecting algorithms here.
        """
        uav_position = self.uav.get_position()
        user_positions = self.user_manager.get_user_positions()
        
        # Resolve strategies: prefer call arguments, otherwise defaults
        bm = beamforming_method or getattr(self, 'default_beamforming_method', 'mrt')
        ps = power_strategy or getattr(self, 'default_power_strategy', 'proportional')

        # print(f"uav_env power_strategy: {ps}")
        # print(f"uav_env beamforming_method: {bm}")

        # Use signal processor's unified interface
        # Algorithm selection is handled in the SignalProcessor (Utils layer)
        total_throughput = self.signal_processor.calculate_system_throughput(
            uav_position=uav_position,
            user_positions=user_positions,
            num_antennas=self.num_antennas,
            total_power_constraint=self.transmit_power,
            channel_model=self.channel_model,
            beamforming_method=bm,
            power_strategy=ps
        )
        
        # Update user throughput history (for info only)
        individual_throughputs = self.signal_processor.get_last_individual_throughputs()
        for i, user in enumerate(self.user_manager.users):
            if i < len(individual_throughputs):
                user.add_throughput(individual_throughputs[i])
        
        return total_throughput
    
    def _calculate_reward(self, current_throughput: float) -> float:
        """
        Calculate reward using the configured reward calculator.
        
        Reward mechanism belongs to Environment layer, but uses configurable parameters.
        
        Args:
            current_throughput: Current step throughput
            
        Returns:
            Calculated reward value
        """
        # Get individual user throughputs
        individual_throughputs = self.signal_processor.get_last_individual_throughputs()
        
        # Calculate reward using the reward calculator
        reward, reward_breakdown = self.reward_calculator.calculate_reward(
            current_throughput=current_throughput,
            uav_position=self.uav.get_position(),
            end_position=self.end_position,
            user_individual_throughputs=individual_throughputs,
            time_step=self.time_step
        )
        
        # Store breakdown for info
        self._last_reward_breakdown = reward_breakdown
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation - simplified and focused."""
        uav_pos = self.uav.get_position()
        end_pos = self.end_position
        
        # Remaining time ratio (normalized)
        remaining_time_ratio = (self.flight_time - self.current_time) / self.flight_time
        
        # Signal quality indicators for each user
        signal_quality_indicators = self._get_signal_quality_indicators()
        
        # Current throughput
        current_throughput = self.episode_throughput_history[-1] if self.episode_throughput_history else 0.0
        
        # Combine observation components
        observation = np.concatenate([
            uav_pos,                    # UAV position (3)
            end_pos,                    # End position (3)
            [remaining_time_ratio],     # Remaining time ratio (1)
            signal_quality_indicators,  # Signal quality per user (num_users)
            [current_throughput]        # Current throughput (1)
        ], dtype=np.float32)
        
        return observation
    
    @property
    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        return max(0.0, self.flight_time - self.current_time)
    
    def _get_signal_quality_indicators(self) -> np.ndarray:
        """
        Get signal quality indicators for each user.
        
        Returns:
            Array of signal quality indicators for each user
        """
        uav_position = self.uav.get_position()
        user_positions = self.user_manager.get_user_positions()
        
        signal_indicators = []
        for user_pos in user_positions:
            # Calculate channel coefficient for this user
            channel_coeff = self.channel_model.calculate_channel_coefficient(
                uav_position, user_pos
            )
            
            # Use channel magnitude as signal quality indicator
            signal_strength = np.linalg.norm(channel_coeff)
            
            # Normalize to [0, 1] range
            signal_indicator = np.clip(signal_strength * 10000.0, 0.0, 1.0)
            signal_indicators.append(signal_indicator)
        
        return np.array(signal_indicators, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment information including reward breakdown."""
        # Basic environment information
        info = {
            'current_step': self.current_step,
            'current_time': self.current_time,
            'total_throughput': self.total_throughput,
            'throughput': self.episode_throughput_history[-1] if self.episode_throughput_history else 0.0,
            'uav_position': self.uav.get_position().tolist(),
            'user_positions': self.user_manager.get_user_positions().tolist(),
            'end_position': self.end_position.tolist(),
            'signal_indicators': self._get_signal_quality_indicators().tolist(),
            'remaining_time': self.remaining_time,
            'individual_throughputs': self.signal_processor.get_last_individual_throughputs(),
            'beamforming_method': getattr(self, 'default_beamforming_method', 'mrt'),
            'power_strategy': getattr(self, 'default_power_strategy', 'proportional')
        }
        
        # Reward breakdown information (for debugging/analysis)
        if hasattr(self, '_last_reward_breakdown'):
            info['reward_breakdown'] = self._last_reward_breakdown
        
        # Reward configuration (for transparency)
        info['reward_config'] = self.reward_config.to_dict()
        
        return info
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        uav_position = self.uav.get_position()
        
        # Check if UAV reached end position (using configured tolerance)
        distance_to_end = np.linalg.norm(uav_position - self.end_position)
        if distance_to_end < self.reward_config.end_position_tolerance:
            return True
        
        # Check if UAV is out of bounds (hard constraint)
        if (uav_position[0] < 0 or uav_position[0] > self.env_size[0] or
            uav_position[1] < 0 or uav_position[1] > self.env_size[1]):
            return True
        
        return False
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        return None
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for discrete action."""
        action_names = {
            0: "East", 1: "South", 2: "West", 3: "North", 4: "Hover"
        }
        return action_names.get(action, "Unknown")
    
    # Simple getter methods (no complex analysis)
    def get_trajectory(self) -> np.ndarray:
        """Get UAV trajectory."""
        return self.uav.get_trajectory()
    
    def get_user_positions(self) -> np.ndarray:
        """Get user positions."""
        return self.user_manager.get_user_positions()
    
    def get_total_throughput(self) -> float:
        """Get total throughput for the episode."""
        return self.total_throughput
    
    def get_throughput_history(self) -> List[float]:
        """Get throughput history for the current episode."""
        return self.episode_throughput_history
    
    def __repr__(self) -> str:
        return (f"UAVEnvironment(users={self.num_users}, antennas={self.num_antennas})")
