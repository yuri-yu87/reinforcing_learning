"""
Mission Planning and Strategy Module

This module contains business logic and decision-making strategies
that were previously embedded in the environment layer.

Responsibilities:
- Mission phase management
- User visiting strategies
- Target selection algorithms
- Time management strategies
- Emergency handling logic

This follows the architectural principle of separating strategic
decision-making from environment simulation.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass


class MissionPhase(Enum):
    """Mission phase enumeration."""
    USER_SELECTION = "user_selection"
    USER_VISITING = "user_visiting"
    TARGET_RUSH = "target_rush"
    EMERGENCY = "emergency"


@dataclass
class MissionState:
    """Current mission state information."""
    current_phase: MissionPhase
    target_user: Optional[int]
    visited_users: set
    phase_start_time: float
    emergency_mode: bool
    user_visit_history: Dict[int, List[float]]
    user_visit_counts: Dict[int, int]


class UserVisitStrategy:
    """Strategies for user visiting behavior."""
    
    @staticmethod
    def nearest_first(uav_position: np.ndarray, 
                     user_positions: np.ndarray, 
                     visited_users: set) -> Optional[int]:
        """Select nearest unvisited user."""
        unvisited = [i for i in range(len(user_positions)) if i not in visited_users]
        if not unvisited:
            return None
        
        distances = [np.linalg.norm(uav_position - user_positions[i]) for i in unvisited]
        nearest_idx = unvisited[np.argmin(distances)]
        return nearest_idx
    
    @staticmethod
    def farthest_first(uav_position: np.ndarray, 
                      user_positions: np.ndarray, 
                      visited_users: set) -> Optional[int]:
        """Select farthest unvisited user (for better coverage)."""
        unvisited = [i for i in range(len(user_positions)) if i not in visited_users]
        if not unvisited:
            return None
        
        distances = [np.linalg.norm(uav_position - user_positions[i]) for i in unvisited]
        farthest_idx = unvisited[np.argmax(distances)]
        return farthest_idx
    
    @staticmethod
    def time_efficient(uav_position: np.ndarray, 
                      user_positions: np.ndarray, 
                      visited_users: set,
                      end_position: np.ndarray,
                      remaining_time: float,
                      max_speed: float) -> Optional[int]:
        """Select user considering time efficiency."""
        unvisited = [i for i in range(len(user_positions)) if i not in visited_users]
        if not unvisited:
            return None
        
        best_user = None
        best_score = float('-inf')
        
        for user_idx in unvisited:
            user_pos = user_positions[user_idx]
            
            # Calculate time cost: UAV -> User -> End
            time_to_user = np.linalg.norm(uav_position - user_pos) / max_speed
            time_user_to_end = np.linalg.norm(user_pos - end_position) / max_speed
            total_time = time_to_user + time_user_to_end
            
            # Score based on time efficiency (prefer shorter total time)
            if total_time < remaining_time:
                score = 1.0 / total_time  # Higher score for shorter time
                if score > best_score:
                    best_score = score
                    best_user = user_idx
        
        return best_user if best_user is not None else unvisited[0]


class MissionPlanner:
    """
    Mission planning and strategic decision making.
    
    This class handles all strategic logic that was previously
    embedded in the environment layer, following the principle
    of separating strategy from environment simulation.
    """
    
    def __init__(self, 
                 num_users: int,
                 visit_strategy: str = 'nearest_first',
                 visit_completion_distance: float = 8.0,
                 visit_completion_signal: float = 0.5,
                 emergency_time_threshold: float = 1.5):
        """
        Initialize mission planner.
        
        Args:
            num_users: Number of users in the mission
            visit_strategy: User visiting strategy ('nearest_first', 'farthest_first', 'time_efficient')
            visit_completion_distance: Distance threshold for visit completion
            visit_completion_signal: Signal quality threshold for visit completion
            emergency_time_threshold: Time multiplier for emergency mode activation
        """
        self.num_users = num_users
        self.visit_strategy = visit_strategy
        self.visit_completion_distance = visit_completion_distance
        self.visit_completion_signal = visit_completion_signal
        self.emergency_time_threshold = emergency_time_threshold
        
        # Strategy function mapping
        self.strategy_functions = {
            'nearest_first': UserVisitStrategy.nearest_first,
            'farthest_first': UserVisitStrategy.farthest_first,
            'time_efficient': UserVisitStrategy.time_efficient
        }
        
        self.reset_mission()
    
    def reset_mission(self):
        """Reset mission state for new episode."""
        self.mission_state = MissionState(
            current_phase=MissionPhase.USER_SELECTION,
            target_user=None,
            visited_users=set(),
            phase_start_time=0.0,
            emergency_mode=False,
            user_visit_history={i: [] for i in range(self.num_users)},
            user_visit_counts={i: 0 for i in range(self.num_users)}
        )
    
    def update_mission_state(self, 
                           uav_position: np.ndarray,
                           user_positions: np.ndarray,
                           end_position: np.ndarray,
                           current_time: float,
                           remaining_time: float,
                           max_speed: float,
                           signal_qualities: np.ndarray) -> Dict[str, Any]:
        """
        Update mission state and return strategic decisions.
        
        Args:
            uav_position: Current UAV position
            user_positions: Array of user positions
            end_position: Mission end position
            current_time: Current mission time
            remaining_time: Remaining mission time
            max_speed: UAV maximum speed
            signal_qualities: Signal quality indicators for each user
            
        Returns:
            Dictionary containing strategic decisions and state updates
        """
        # Update user visit tracking
        self._update_user_visits(uav_position, user_positions, current_time)
        
        # Check for emergency mode
        self._check_emergency_mode(uav_position, end_position, remaining_time, max_speed)
        
        # Update mission phase
        old_phase = self.mission_state.current_phase
        self._update_mission_phase(uav_position, user_positions, end_position, 
                                 current_time, remaining_time, max_speed, signal_qualities)
        
        # Generate strategic recommendations
        recommendations = self._generate_recommendations(
            uav_position, user_positions, end_position, remaining_time, max_speed
        )
        
        return {
            'mission_state': self.mission_state,
            'phase_changed': old_phase != self.mission_state.current_phase,
            'recommendations': recommendations,
            'strategic_priority': self._get_strategic_priority(),
            'time_pressure': self._calculate_time_pressure(remaining_time, 
                                                         uav_position, end_position, max_speed)
        }
    
    def _update_user_visits(self, uav_position: np.ndarray, user_positions: np.ndarray, current_time: float):
        """Update user visit tracking."""
        for user_idx in range(self.num_users):
            user_pos = user_positions[user_idx]
            distance = np.linalg.norm(uav_position - user_pos)
            
            # Check if visiting this user
            if distance < self.visit_completion_distance:
                # Check if this is a new visit (avoid multiple counts for same visit)
                if (not self.mission_state.user_visit_history[user_idx] or 
                    current_time - self.mission_state.user_visit_history[user_idx][-1] > 10.0):
                    
                    self.mission_state.user_visit_history[user_idx].append(current_time)
                    self.mission_state.user_visit_counts[user_idx] += 1
                    self.mission_state.visited_users.add(user_idx)
    
    def _check_emergency_mode(self, uav_position: np.ndarray, end_position: np.ndarray, 
                            remaining_time: float, max_speed: float):
        """Check if emergency mode should be activated."""
        distance_to_end = np.linalg.norm(uav_position - end_position)
        min_time_needed = distance_to_end / max_speed
        
        if remaining_time < min_time_needed * self.emergency_time_threshold:
            self.mission_state.emergency_mode = True
            self.mission_state.current_phase = MissionPhase.EMERGENCY
        else:
            self.mission_state.emergency_mode = False
    
    def _update_mission_phase(self, uav_position: np.ndarray, user_positions: np.ndarray, 
                            end_position: np.ndarray, current_time: float, remaining_time: float,
                            max_speed: float, signal_qualities: np.ndarray):
        """Update mission phase based on current state."""
        if self.mission_state.emergency_mode:
            return  # Stay in emergency mode
        
        if self.mission_state.current_phase == MissionPhase.USER_SELECTION:
            # Select target user if none selected
            if self.mission_state.target_user is None:
                self.mission_state.target_user = self._select_target_user(
                    uav_position, user_positions, end_position, remaining_time, max_speed
                )
                if self.mission_state.target_user is not None:
                    self.mission_state.current_phase = MissionPhase.USER_VISITING
                    self.mission_state.phase_start_time = current_time
        
        elif self.mission_state.current_phase == MissionPhase.USER_VISITING:
            # Check if current user visit is complete
            if (self.mission_state.target_user is not None and 
                self._is_user_visit_complete(self.mission_state.target_user, 
                                           uav_position, user_positions, signal_qualities)):
                
                self.mission_state.visited_users.add(self.mission_state.target_user)
                self.mission_state.target_user = None
                
                # Check if all users visited
                if len(self.mission_state.visited_users) >= self.num_users:
                    self.mission_state.current_phase = MissionPhase.TARGET_RUSH
                    self.mission_state.phase_start_time = current_time
                else:
                    self.mission_state.current_phase = MissionPhase.USER_SELECTION
                    self.mission_state.phase_start_time = current_time
    
    def _select_target_user(self, uav_position: np.ndarray, user_positions: np.ndarray,
                          end_position: np.ndarray, remaining_time: float, max_speed: float) -> Optional[int]:
        """Select target user based on strategy."""
        strategy_func = self.strategy_functions.get(self.visit_strategy, 
                                                   UserVisitStrategy.nearest_first)
        
        if self.visit_strategy == 'time_efficient':
            return strategy_func(uav_position, user_positions, self.mission_state.visited_users,
                               end_position, remaining_time, max_speed)
        else:
            return strategy_func(uav_position, user_positions, self.mission_state.visited_users)
    
    def _is_user_visit_complete(self, user_id: int, uav_position: np.ndarray, 
                              user_positions: np.ndarray, signal_qualities: np.ndarray) -> bool:
        """Check if user visit is complete."""
        if user_id >= len(user_positions) or user_id >= len(signal_qualities):
            return False
        
        distance = np.linalg.norm(uav_position - user_positions[user_id])
        signal_quality = signal_qualities[user_id]
        
        # Relaxed completion criteria (OR logic)
        distance_ok = distance < self.visit_completion_distance
        signal_ok = signal_quality > self.visit_completion_signal
        moderate_both = distance < (self.visit_completion_distance * 1.5) and signal_quality > (self.visit_completion_signal * 0.6)
        
        return distance_ok or signal_ok or moderate_both
    
    def _generate_recommendations(self, uav_position: np.ndarray, user_positions: np.ndarray,
                                end_position: np.ndarray, remaining_time: float, max_speed: float) -> Dict[str, Any]:
        """Generate strategic recommendations for the agent."""
        recommendations = {
            'priority_target': None,
            'recommended_action': 'explore',
            'urgency_level': 'normal',
            'risk_assessment': 'low'
        }
        
        if self.mission_state.current_phase == MissionPhase.USER_SELECTION:
            recommendations['priority_target'] = 'select_user'
            recommendations['recommended_action'] = 'move_to_nearest_unvisited'
            
        elif self.mission_state.current_phase == MissionPhase.USER_VISITING:
            if self.mission_state.target_user is not None:
                recommendations['priority_target'] = f'user_{self.mission_state.target_user}'
                recommendations['recommended_action'] = 'approach_target_user'
            
        elif self.mission_state.current_phase in [MissionPhase.TARGET_RUSH, MissionPhase.EMERGENCY]:
            recommendations['priority_target'] = 'end_position'
            recommendations['recommended_action'] = 'move_to_end'
            recommendations['urgency_level'] = 'high' if self.mission_state.emergency_mode else 'medium'
        
        # Risk assessment
        time_pressure = self._calculate_time_pressure(remaining_time, uav_position, end_position, max_speed)
        if time_pressure > 0.8:
            recommendations['risk_assessment'] = 'high'
        elif time_pressure > 0.5:
            recommendations['risk_assessment'] = 'medium'
        
        return recommendations
    
    def _get_strategic_priority(self) -> str:
        """Get current strategic priority."""
        if self.mission_state.emergency_mode:
            return 'mission_completion'
        elif self.mission_state.current_phase == MissionPhase.USER_VISITING:
            return 'user_service'
        elif self.mission_state.current_phase == MissionPhase.TARGET_RUSH:
            return 'mission_completion'
        else:
            return 'exploration'
    
    def _calculate_time_pressure(self, remaining_time: float, uav_position: np.ndarray,
                               end_position: np.ndarray, max_speed: float) -> float:
        """Calculate time pressure ratio (0=no pressure, 1=critical)."""
        distance_to_end = np.linalg.norm(uav_position - end_position)
        min_time_needed = distance_to_end / max_speed
        
        if remaining_time <= 0:
            return 1.0
        
        time_pressure = min_time_needed / remaining_time
        return float(np.clip(time_pressure, 0.0, 1.0))
    
    def get_mission_summary(self) -> Dict[str, Any]:
        """Get summary of mission state and progress."""
        total_visits = sum(self.mission_state.user_visit_counts.values())
        unique_users_visited = len(self.mission_state.visited_users)
        
        return {
            'current_phase': self.mission_state.current_phase.value,
            'target_user': self.mission_state.target_user,
            'users_visited': unique_users_visited,
            'total_users': self.num_users,
            'completion_ratio': unique_users_visited / self.num_users,
            'total_visits': total_visits,
            'visit_counts': self.mission_state.user_visit_counts.copy(),
            'emergency_mode': self.mission_state.emergency_mode,
            'strategy': self.visit_strategy
        }
