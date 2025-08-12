import numpy as np
from typing import Tuple, List, Optional


class UAV:
    """
    UAV class representing an aerial base station with multiple antennas.
    
    Attributes:
        position (np.ndarray): Current 3D position (x, y, z)
        velocity (np.ndarray): Current velocity vector (vx, vy, vz)
        num_antennas (int): Number of transmit antennas
        max_speed (float): Maximum speed in m/s
        min_speed (float): Minimum speed in m/s
        height (float): Fixed flight height
        transmit_power (float): Total transmit power budget in Watts
    """
    
    def __init__(self, 
                 start_position: Tuple[float, float, float] = (0, 0, 50),
                 num_antennas: int = 8,
                 max_speed: float = 30.0,
                 min_speed: float = 10.0,
                 transmit_power: float = 0.5,
                 env_bounds: Optional[Tuple[float, float, float]] = None):
        """
        Initialize UAV with given parameters.
        
        Args:
            start_position: Initial position (x, y, z) in meters
            num_antennas: Number of transmit antennas
            max_speed: Maximum speed in m/s
            min_speed: Minimum speed in m/s
            transmit_power: Total transmit power budget in Watts
            env_bounds: Environment bounds (x_max, y_max, z_max)
        """
        self.position = np.array(start_position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32) #[0,0,0] # 3D velocity vector
        self.num_antennas = num_antennas
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.height = start_position[2] # Fixed height
        self.transmit_power = transmit_power
        
        # Environment bounds for boundary constraints
        self.env_bounds = env_bounds if env_bounds is not None else (100, 100, 50)
        
        # History for trajectory tracking
        self.trajectory = [self.position.copy()] # record the trajectory of the UAV
        
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.velocity.copy()
    
    def update_position(self, position: np.ndarray) -> None:
        """Set and Record UAV position."""
        self.position = position.copy()
        self.position[2] = self.height  # Maintain fixed height
        self.trajectory.append(self.position.copy())
    
    def move(self, direction: np.ndarray, speed: float, time_step: float = 1.0) -> bool:
        """
        Move UAV in the given direction with specified speed for given time step.
        
        Args:
            direction: Normalized direction vector (2D: x, y)
            speed: Speed in m/s
            time_step: Time step duration in seconds
            
        Returns:
            bool: True if movement is valid, False otherwise
        """
        # Ensure speed is within bounds
        speed = np.clip(speed, self.min_speed, self.max_speed)
        
        # Normalize direction vector
        direction_2d = direction[:2]
        if np.linalg.norm(direction_2d) > 0:
            direction_2d = direction_2d / np.linalg.norm(direction_2d) #Euclidean norm
        
        # Calculate new position based on time step
        new_position = self.position.copy().astype(np.float64)  # Ensure float64 type
        new_position[:2] += direction_2d * speed * time_step  # Distance = speed * time
        
        # Apply boundary constraints
        new_position[0] = np.clip(new_position[0], 0, self.env_bounds[0])  # X bounds
        new_position[1] = np.clip(new_position[1], 0, self.env_bounds[1])  # Y bounds
        new_position[2] = self.height  # Maintain fixed height
        
        # Update position
        self.update_position(new_position)
        self.velocity = np.concatenate([direction_2d * speed, [0]])
        
        return True
    
    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance to a given point."""
        return np.linalg.norm(self.position - point)
    
    def get_trajectory(self) -> np.ndarray:
        """Get complete trajectory history."""
        return np.array(self.trajectory)
    
    def reset(self, start_position: Optional[Tuple[float, float, float]] = None) -> None:
        """Reset UAV to start position."""
        if start_position is None:
            start_position = (0, 0, self.height)
        
        self.position = np.array(start_position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.trajectory = [self.position.copy()]
    
    def __repr__(self) -> str:
        return f"UAV(position={self.position}, antennas={self.num_antennas}, power={self.transmit_power}W)" 