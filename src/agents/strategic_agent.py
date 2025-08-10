"""
Strategic Agent with Integrated Mission Planner

This module demonstrates how to integrate MissionPlanner at the Agent layer,
following the architectural principle that decision-making logic belongs 
to Agent/Policy layer, not Environment layer.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from agents.mission_planner import MissionPlanner
from environment.uav_env import UAVEnvironment


class StrategicAgent:
    """
    Agent that integrates MissionPlanner for strategic decision making.
    
    This shows the correct architectural pattern:
    - Environment provides observations and rewards
    - Agent (this class) uses MissionPlanner for strategy
    - Agent combines strategic recommendations with learned policy
    """
    
    def __init__(self, 
                 num_users: int = 2,
                 mission_strategy: str = 'time_efficient',
                 use_strategic_guidance: bool = True,
                 strategic_weight: float = 0.3):
        """
        Initialize strategic agent.
        
        Args:
            num_users: Number of users in environment
            mission_strategy: Strategy for MissionPlanner
            use_strategic_guidance: Whether to use mission planner guidance
            strategic_weight: Weight for strategic recommendations (0=ignore, 1=follow completely)
        """
        self.num_users = num_users
        self.use_strategic_guidance = use_strategic_guidance
        self.strategic_weight = strategic_weight
        
        # Initialize MissionPlanner (Agent layer responsibility)
        if self.use_strategic_guidance:
            self.mission_planner = MissionPlanner(
                num_users=num_users,
                visit_strategy=mission_strategy,
                visit_completion_distance=8.0,
                visit_completion_signal=0.5,
                emergency_time_threshold=1.5
            )
        else:
            self.mission_planner = None
        
        # Action space mapping (should match environment's action space)
        self.action_meanings = {
            0: "move_forward",
            1: "move_backward", 
            2: "move_left",
            3: "move_right",
            4: "hover"
        }
    
    def reset(self):
        """Reset agent for new episode."""
        if self.mission_planner:
            self.mission_planner.reset_mission()
    
    def get_action(self, 
                   observation: np.ndarray,
                   info: Dict[str, Any],
                   learned_policy_action: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Get action combining strategic guidance with learned policy.
        
        Args:
            observation: Environment observation
            info: Environment info dict
            learned_policy_action: Action from learned policy (e.g., RL network)
            
        Returns:
            Tuple of (final_action, agent_info)
        """
        agent_info = {
            'strategic_guidance_used': False,
            'mission_phase': None,
            'target_user': None,
            'recommended_action': None,
            'final_action_source': 'random'
        }
        
        # Get strategic guidance if enabled
        strategic_action = None
        if self.use_strategic_guidance and self.mission_planner:
            strategic_action, mission_info = self._get_strategic_action(info)
            agent_info.update(mission_info)
            agent_info['strategic_guidance_used'] = True
        
        # Combine strategic guidance with learned policy
        if learned_policy_action is not None:
            # Have learned policy - combine with strategic guidance
            final_action = self._combine_actions(learned_policy_action, strategic_action, info)
            agent_info['final_action_source'] = 'combined'
        elif strategic_action is not None:
            # Only strategic guidance available
            final_action = strategic_action
            agent_info['final_action_source'] = 'strategic'
        else:
            # Fallback to random action
            final_action = np.random.choice(len(self.action_meanings))
            agent_info['final_action_source'] = 'random'
        
        agent_info['final_action'] = final_action
        agent_info['action_meaning'] = self.action_meanings.get(final_action, 'unknown')
        
        return final_action, agent_info
    
    def _get_strategic_action(self, info: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Any]]:
        """Get strategic action recommendation from MissionPlanner."""
        # Extract required information from environment info
        uav_position = np.array(info['uav_position'])
        user_positions = np.array(info['user_positions'])
        end_position = np.array(info['end_position'])
        current_time = info['current_time']
        remaining_time = info['remaining_time']
        signal_indicators = np.array(info['signal_indicators'])
        
        # Update mission planner state
        mission_update = self.mission_planner.update_mission_state(
            uav_position=uav_position,
            user_positions=user_positions,
            end_position=end_position,
            current_time=current_time,
            remaining_time=remaining_time,
            max_speed=30.0,  # Should match environment configuration
            signal_qualities=signal_indicators
        )
        
        # Convert strategic recommendations to action
        strategic_action = self._convert_recommendation_to_action(
            mission_update['recommendations'],
            uav_position,
            user_positions,
            end_position
        )
        
        mission_info = {
            'mission_phase': mission_update['mission_state'].current_phase.value,
            'target_user': mission_update['mission_state'].target_user,
            'recommended_action': mission_update['recommendations']['recommended_action'],
            'urgency_level': mission_update['recommendations']['urgency_level'],
            'time_pressure': mission_update['time_pressure'],
            'mission_summary': self.mission_planner.get_mission_summary()
        }
        
        return strategic_action, mission_info
    
    def _convert_recommendation_to_action(self,
                                        recommendations: Dict[str, Any],
                                        uav_position: np.ndarray,
                                        user_positions: np.ndarray,
                                        end_position: np.ndarray) -> Optional[int]:
        """Convert strategic recommendations to specific actions."""
        rec_action = recommendations.get('recommended_action', 'explore')
        priority_target = recommendations.get('priority_target')
        
        # Determine target position based on recommendation
        target_position = None
        
        if rec_action == 'move_to_end' or priority_target == 'end_position':
            target_position = end_position
        elif rec_action in ['approach_target_user', 'move_to_nearest_unvisited']:
            if priority_target and priority_target.startswith('user_'):
                try:
                    user_idx = int(priority_target.split('_')[1])
                    if 0 <= user_idx < len(user_positions):
                        target_position = user_positions[user_idx]
                except (ValueError, IndexError):
                    pass
            
            # Fallback to nearest user
            if target_position is None and len(user_positions) > 0:
                distances = [np.linalg.norm(uav_position - user_pos) for user_pos in user_positions]
                nearest_idx = np.argmin(distances)
                target_position = user_positions[nearest_idx]
        
        # Convert target position to action
        if target_position is not None:
            return self._position_to_action(uav_position, target_position)
        
        # Default action for exploration
        return np.random.choice([0, 2, 3])  # forward, left, right
    
    def _position_to_action(self, current_pos: np.ndarray, target_pos: np.ndarray) -> int:
        """Convert target position to action."""
        direction = target_pos - current_pos
        
        # Simple direction-based action selection
        if abs(direction[0]) > abs(direction[1]):
            # Primary movement in x-direction
            if direction[0] > 0:
                return 0  # move_forward (assuming forward is +x)
            else:
                return 1  # move_backward
        else:
            # Primary movement in y-direction  
            if direction[1] > 0:
                return 3  # move_right (assuming right is +y)
            else:
                return 2  # move_left
    
    def _combine_actions(self, 
                        learned_action: int, 
                        strategic_action: Optional[int],
                        info: Dict[str, Any]) -> int:
        """Combine learned policy action with strategic guidance."""
        if strategic_action is None:
            return learned_action
        
        # Simple weighted combination based on urgency
        urgency_map = {'low': 0.1, 'normal': 0.3, 'medium': 0.5, 'high': 0.8}
        
        # Get urgency from info if available
        urgency = 'normal'
        if 'reward_breakdown' in info and 'time_pressure' in info:
            time_pressure = info.get('time_pressure', 0.0)
            if time_pressure > 0.8:
                urgency = 'high'
            elif time_pressure > 0.5:
                urgency = 'medium'
        
        strategic_weight = urgency_map.get(urgency, 0.3)
        
        # Use strategic action if random threshold is met
        if np.random.random() < strategic_weight:
            return strategic_action
        else:
            return learned_action


class RandomPolicyWithStrategy(StrategicAgent):
    """Example implementation: Random policy with strategic guidance."""
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Get action using random policy with strategic guidance."""
        # Generate random policy action
        random_action = np.random.choice(len(self.action_meanings))
        
        # Combine with strategic guidance
        return super().get_action(observation, info, learned_policy_action=random_action)


# Example usage for integration with training
def create_strategic_agent(env_config: Dict[str, Any]) -> StrategicAgent:
    """Factory function to create strategic agent based on environment config."""
    return StrategicAgent(
        num_users=env_config.get('num_users', 2),
        mission_strategy=env_config.get('mission_strategy', 'time_efficient'),
        use_strategic_guidance=env_config.get('use_strategic_guidance', True),
        strategic_weight=env_config.get('strategic_weight', 0.3)
    )
