import numpy as np
from typing import List, Tuple, Optional


class GroundUser:
    """
    Ground user class representing a user in the 3D environment.
    
    Attributes:
        position (np.ndarray): 3D position (x, y, z=0 for ground users)
        user_id (int): Unique user identifier
        throughput_history (List[float]): Historical throughput values
    """
    
    def __init__(self, position: Tuple[float, float, float], user_id: int):
        """
        Initialize ground user.
        
        Args:
            position: 3D position (x, y, z)
            user_id: Unique user identifier
        """
        self.position = np.array(position, dtype=np.float32)
        self.position[2] = 0  # Ground users are at z=0
        self.user_id = user_id
        self.throughput_history = []
        
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.position.copy()
    
    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance to a given point."""
        return np.linalg.norm(self.position - point)
    
    def add_throughput(self, throughput: float) -> None:
        """Add throughput measurement to history."""
        self.throughput_history.append(throughput)
    
    def get_average_throughput(self) -> float:
        """Get average throughput over history."""
        if not self.throughput_history:
            return 0.0
        return np.mean(self.throughput_history)
    
    def reset_throughput_history(self) -> None:
        """Reset throughput history."""
        self.throughput_history = []
    
    def __repr__(self) -> str:
        return f"User{self.user_id}(position={self.position})"


class UserManager:
    """
    Manager class for multiple ground users.
    
    Attributes:
        users (List[GroundUser]): List of ground users
        num_users (int): Number of users
    """
    
    def __init__(self, num_users: int = 2):
        """
        Initialize user manager.
        
        Args:
            num_users: Number of ground users to create
        """
        self.num_users = num_users
        self.users = []
        
    def generate_random_users(self, 
                            x_range: Tuple[float, float] = (0, 100),
                            y_range: Tuple[float, float] = (0, 100),
                            seed: Optional[int] = None) -> None:
        """
        Generate random user positions.
        
        Args:
            x_range: Range for x coordinates (min, max)
            y_range: Range for y coordinates (min, max)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.users = []
        for i in range(self.num_users):
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = 0  # Ground users are at z=0
            
            user = GroundUser((x, y, z), user_id=i)
            self.users.append(user)
    
    def set_fixed_users(self, positions: np.ndarray) -> None:
        """
        Set fixed user positions.
        
        Args:
            positions: Array of user positions with shape (num_users, 3)
        """
        if len(positions) != self.num_users:
            raise ValueError(f"Expected {self.num_users} positions, got {len(positions)}")
            
        self.users = []
        for i, pos in enumerate(positions):
            user = GroundUser((pos[0], pos[1], pos[2]), user_id=i)
            self.users.append(user)
    
    def set_user_positions(self, positions: np.ndarray) -> None:
        """
        Set user positions (alias for set_fixed_users for consistency).
        
        Args:
            positions: Array of user positions with shape (num_users, 3)
        """
        self.set_fixed_users(positions)
    
    def get_user_positions(self) -> np.ndarray:
        """Get positions of all users as numpy array."""
        return np.array([user.get_position() for user in self.users])
    
    def get_user_by_id(self, user_id: int) -> Optional[GroundUser]:
        """Get user by ID."""
        for user in self.users:
            if user.user_id == user_id:
                return user
        return None
    
    def add_throughput_to_user(self, user_id: int, throughput: float) -> None:
        """Add throughput measurement to specific user."""
        user = self.get_user_by_id(user_id)
        if user:
            user.add_throughput(throughput)
    
    def get_total_throughput(self) -> float:
        """Get total throughput across all users."""
        return sum(user.get_average_throughput() for user in self.users)
    
    def reset_all_throughput_history(self) -> None:
        """Reset throughput history for all users."""
        for user in self.users:
            user.reset_throughput_history()
    
    def get_minimum_distance_to_users(self, point: np.ndarray) -> float:
        """Get minimum distance from point to any user."""
        distances = [user.distance_to(point) for user in self.users]
        return min(distances) if distances else float('inf')
    
    def __len__(self) -> int:
        return len(self.users)
    
    def __getitem__(self, index: int) -> GroundUser:
        return self.users[index]
    
    def __repr__(self) -> str:
        return f"UserManager({len(self.users)} users)" 