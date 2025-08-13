"""
高级终点引导奖励系统
针对6阶段课程学习进行精细调优，确保UAV在访问完用户后强烈倾向于前往终点
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class AdvancedEndpointGuidanceConfig:
    """
    高级终点引导配置
    专为解决"UAV访问完用户但不去终点"的问题设计
    """
    
    # === 基础奖励权重 ===
    w_throughput_base: float = 100.0
    w_movement_bonus: float = 15.0
    
    # === 用户访问奖励（递减策略）===
    B_user_visit_base: float = 3000.0      # 基础单用户访问奖励
    B_user_visit_decay: float = 0.8        # 后续用户奖励衰减因子
    B_all_users_visited: float = 8000.0    # 全用户访问完成奖励
    B_sequential_bonus: float = 2000.0     # 顺序访问奖励
    
    # === 终点引导奖励（大幅增强）===
    B_reach_end_base: float = 6000.0           # 基础到达终点奖励
    B_mission_complete: float = 15000.0         # 完整任务完成奖励
    
    # === 动态终点引导机制 ===
    w_end_approach_base: float = 200.0          # 基础终点接近权重
    w_end_urgency_multiplier: float = 5.0       # 访问完用户后的紧迫度倍数
    w_end_progress_base: float = 100.0          # 基础终点进展权重
    w_end_progress_multiplier: float = 8.0      # 访问完用户后的进展倍数
    
    # === 智能惩罚机制 ===
    penalty_incomplete_mission: float = 2000.0  # 未完成任务的持续惩罚
    penalty_wasted_time: float = 50.0           # 访问完用户后的时间浪费惩罚
    penalty_distance_from_end: float = 100.0    # 访问完用户后远离终点的惩罚
    
    # === 服务参数 ===
    user_service_radius: float = 60.0
    close_to_user_threshold: float = 80.0
    end_position_tolerance: float = 25.0
    user_visit_time_threshold: float = 0.8
    
    # === 时间约束 ===
    min_flight_time: float = 200.0
    max_flight_time: float = 300.0
    time_step: float = 0.1
    
    # === 引导阈值 ===
    end_guidance_activation_distance: float = 150.0   # 终点引导激活距离
    strong_guidance_activation_distance: float = 80.0  # 强引导激活距离
    
    # === 其他参数 ===
    stagnation_threshold: float = 1.0
    stagnation_time_window: float = 3.0
    w_stagnation: float = 5.0
    w_oob: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'B_user_visit_base': self.B_user_visit_base,
            'B_all_users_visited': self.B_all_users_visited,
            'B_reach_end_base': self.B_reach_end_base,
            'B_mission_complete': self.B_mission_complete,
            'w_end_approach_base': self.w_end_approach_base,
            'w_end_urgency_multiplier': self.w_end_urgency_multiplier,
            'penalty_incomplete_mission': self.penalty_incomplete_mission,
            'user_service_radius': self.user_service_radius,
            'end_position_tolerance': self.end_position_tolerance
        }


class AdvancedEndpointGuidanceCalculator:
    """
    高级终点引导奖励计算器
    实现智能化、自适应的终点引导机制
    """
    
    def __init__(self, config: AdvancedEndpointGuidanceConfig, stage_manager=None):
        self.config = config
        self.stage_manager = stage_manager
        self.reset()
    
    def reset(self):
        """重置计算器状态"""
        # 用户访问跟踪
        self.user_visit_times = {}
        self.user_visited_flags = set()
        self.user_visit_order = []
        
        # 位置和状态跟踪
        self.position_history = []
        self.time_history = []
        self.previous_position = None
        self.last_target_distances = {}
        
        # 终点引导状态
        self.all_users_visited = False
        self.all_users_visited_time = None
        self.end_guidance_activated = False
        self.strong_guidance_activated = False
        
        # 性能跟踪
        self.time_since_all_users_visited = 0.0
        self.mission_completed = False
    
    def calculate_reward(self, 
                        uav_position: np.ndarray,
                        end_position: np.ndarray,
                        user_positions: np.ndarray,
                        user_throughputs: np.ndarray,
                        current_time: float,
                        current_speed: float,
                        env_bounds: tuple,
                        episode_done: bool,
                        reached_end: bool) -> Dict[str, float]:
        """
        高级终点引导奖励计算
        """
        reward_breakdown = {}
        
        # 获取当前阶段配置（如果有stage_manager）
        if self.stage_manager:
            stage_config = self.stage_manager.get_stage_config(self.stage_manager.current_stage)
            effective_user_positions = stage_config['user_positions']
            stage_multipliers = stage_config['reward_multipliers']
        else:
            effective_user_positions = user_positions
            stage_multipliers = {'user_visit': 1.0, 'approach': 1.0, 'completion': 1.0}
        
        # === 1. 基础吞吐量奖励 ===
        throughput_reward = self.config.w_throughput_base * np.sum(user_throughputs)
        reward_breakdown['throughput'] = throughput_reward
        
        # === 2. 移动奖励 ===
        movement_reward = 0.0
        if self.previous_position is not None:
            displacement = np.linalg.norm(uav_position - self.previous_position)
            movement_reward = self.config.w_movement_bonus * min(displacement / 3.0, 1.0)
        reward_breakdown['movement'] = movement_reward
        
        # === 3. 用户访问处理和奖励 ===
        visit_rewards = self._update_user_visits_and_calculate_rewards(
            uav_position, effective_user_positions, current_time, stage_multipliers
        )
        reward_breakdown.update(visit_rewards)
        
        # === 4. 智能终点引导机制 ===
        guidance_rewards = self._calculate_advanced_endpoint_guidance(
            uav_position, end_position, effective_user_positions, current_time
        )
        reward_breakdown.update(guidance_rewards)
        
        # === 5. 进展奖励（自适应） ===
        progress_rewards = self._calculate_adaptive_progress_rewards(
            uav_position, effective_user_positions, end_position
        )
        reward_breakdown.update(progress_rewards)
        
        # === 6. 智能惩罚机制 ===
        penalties = self._calculate_intelligent_penalties(
            uav_position, end_position, env_bounds, current_time
        )
        reward_breakdown.update(penalties)
        
        # === 7. 终端奖励（增强） ===
        if episode_done:
            terminal_rewards = self._calculate_enhanced_terminal_rewards(
                reached_end, current_time, len(effective_user_positions), stage_multipliers
            )
            reward_breakdown.update(terminal_rewards)
        
        # === 8. 计算总奖励 ===
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total'] = total_reward
        
        # === 9. 更新状态 ===
        self._update_internal_state(uav_position, effective_user_positions, end_position, current_time)
        
        return reward_breakdown
    
    def _update_user_visits_and_calculate_rewards(self, uav_position: np.ndarray, 
                                                 user_positions: np.ndarray, current_time: float,
                                                 multipliers: Dict[str, float]) -> Dict[str, float]:
        """更新用户访问并计算递减奖励"""
        rewards = {
            'user_visit_bonus': 0.0
        }
        
        # 只处理有效用户
        for user_id, user_pos in enumerate(user_positions):
            distance = np.linalg.norm(uav_position - user_pos)
            
            # 在服务半径内累积访问时间
            if distance <= self.config.user_service_radius:
                if user_id not in self.user_visit_times:
                    self.user_visit_times[user_id] = 0.0
                self.user_visit_times[user_id] += self.config.time_step
                
                # 检查是否完成访问
                if (self.user_visit_times[user_id] >= self.config.user_visit_time_threshold and
                    user_id not in self.user_visited_flags):
                    
                    self.user_visited_flags.add(user_id)
                    self.user_visit_order.append(user_id)
                    
                    # === 递减访问奖励机制 ===
                    visit_count = len(self.user_visited_flags)
                    
                    # 基础奖励随访问顺序递减
                    decay_factor = self.config.B_user_visit_decay ** (visit_count - 1)
                    base_visit_reward = self.config.B_user_visit_base * decay_factor * multipliers['user_visit']
                    
                    # 顺序奖励（鼓励连续访问）
                    if visit_count > 1:
                        sequential_reward = self.config.B_sequential_bonus * 0.5
                        base_visit_reward += sequential_reward
                    
                    rewards['user_visit_bonus'] += base_visit_reward
                    
                    print(f"🎯 用户{user_id}访问完成！获得{base_visit_reward:.0f}奖励（第{visit_count}个）")
                    
                    # === 全用户访问完成奖励 ===
                    if len(self.user_visited_flags) == len(user_positions):
                        all_users_reward = self.config.B_all_users_visited
                        rewards['user_visit_bonus'] += all_users_reward
                        
                        # 记录完成时间，激活终点引导
                        if not self.all_users_visited:
                            self.all_users_visited = True
                            self.all_users_visited_time = current_time
                            self.end_guidance_activated = True
                            print(f"🏆 全用户访问完成！激活终点引导机制！获得{all_users_reward:.0f}额外奖励")
        
        return rewards
    
    def _calculate_advanced_endpoint_guidance(self, uav_position: np.ndarray, 
                                            end_position: np.ndarray, user_positions: np.ndarray,
                                            current_time: float) -> Dict[str, float]:
        """高级终点引导机制"""
        rewards = {
            'end_approach': 0.0,
            'end_urgency': 0.0,
            'end_magnetism': 0.0,
            'completion_drive': 0.0
        }
        
        distance_to_end = np.linalg.norm(uav_position - end_position)
        
        # === 阶段1：基础终点引导（距离较远时） ===
        if distance_to_end <= self.config.end_guidance_activation_distance:
            base_proximity_factor = (self.config.end_guidance_activation_distance - distance_to_end) / self.config.end_guidance_activation_distance
            base_approach_reward = self.config.w_end_approach_base * base_proximity_factor * 0.2
            
            # 根据用户访问状态调整奖励
            users_visited = len(self.user_visited_flags)
            total_users = len(user_positions)
            
            if users_visited == total_users:
                # 访问完所有用户：全功率终点引导
                rewards['end_approach'] = base_approach_reward * self.config.w_end_urgency_multiplier
                
                # === 阶段2：强引导机制（距离中等时） ===
                if distance_to_end <= self.config.strong_guidance_activation_distance:
                    if not self.strong_guidance_activated:
                        self.strong_guidance_activated = True
                        print("⚡ 激活强终点引导机制！")
                    
                    # 强引导奖励
                    strong_proximity_factor = (self.config.strong_guidance_activation_distance - distance_to_end) / self.config.strong_guidance_activation_distance
                    rewards['end_urgency'] = self.config.w_end_approach_base * strong_proximity_factor * self.config.w_end_urgency_multiplier * 2.0
                
                # === 阶段3：磁吸效应（距离很近时） ===
                if distance_to_end <= 40.0:
                    magnetism_factor = (40.0 - distance_to_end) / 40.0
                    magnetism_reward = self.config.w_end_approach_base * magnetism_factor * magnetism_factor * 10.0
                    rewards['end_magnetism'] = magnetism_reward
                
                # === 持续完成驱动力 ===
                if self.all_users_visited_time is not None:
                    time_since_completion = current_time - self.all_users_visited_time
                    # 时间越长，驱动力越强（避免浪费时间）
                    time_urgency = min(time_since_completion / 20.0, 5.0)  # 最多5倍
                    completion_drive = self.config.w_end_approach_base * time_urgency * 2.0
                    rewards['completion_drive'] = completion_drive
                    
            elif users_visited > 0:
                # 访问了部分用户：中等终点引导
                completion_ratio = users_visited / total_users
                rewards['end_approach'] = base_approach_reward * completion_ratio * 0.3
            else:
                # 没有访问用户：微弱终点引导（避免直接去终点）
                rewards['end_approach'] = base_approach_reward * 0.05
        
        return rewards
    
    def _calculate_adaptive_progress_rewards(self, uav_position: np.ndarray, 
                                           user_positions: np.ndarray, end_position: np.ndarray) -> Dict[str, float]:
        """自适应进展奖励"""
        rewards = {
            'user_progress': 0.0,
            'end_progress': 0.0,
            'super_end_progress': 0.0
        }
        
        # 计算当前距离
        current_distances = {}
        
        # 到未访问用户的距离
        for i, user_pos in enumerate(user_positions):
            if i not in self.user_visited_flags:
                current_distances[f'user_{i}'] = np.linalg.norm(uav_position - user_pos)
        
        # 到终点的距离
        end_distance = np.linalg.norm(uav_position - end_position)
        current_distances['end'] = end_distance
        
        # 计算进展奖励
        for target, current_dist in current_distances.items():
            if target in self.last_target_distances:
                last_dist = self.last_target_distances[target]
                progress = last_dist - current_dist
                
                if progress > 0:  # 距离减少了
                    if target == 'end':
                        # === 终点进展奖励（自适应增强）===
                        base_reward = self.config.w_end_progress_base * progress * 0.3
                        
                        if self.all_users_visited:
                            # 访问完所有用户后，终点进展奖励大幅增强
                            enhanced_reward = base_reward * self.config.w_end_progress_multiplier
                            rewards['end_progress'] = enhanced_reward
                            
                            # 超级进展奖励（距离很近时）
                            if current_dist <= 50.0:
                                super_bonus = self.config.w_end_progress_base * progress * 5.0
                                rewards['super_end_progress'] = super_bonus
                        else:
                            rewards['end_progress'] = base_reward * 0.2
                    else:
                        # 用户进展奖励
                        rewards['user_progress'] += self.config.w_end_progress_base * progress * 0.15
        
        # 更新距离记录
        self.last_target_distances = current_distances
        
        return rewards
    
    def _calculate_intelligent_penalties(self, uav_position: np.ndarray, 
                                       end_position: np.ndarray, env_bounds: tuple,
                                       current_time: float) -> Dict[str, float]:
        """智能惩罚机制"""
        penalties = {
            'oob_penalty': 0.0,
            'stagnation_penalty': 0.0,
            'time_waste_penalty': 0.0,
            'distance_penalty': 0.0,
            'incomplete_mission_penalty': 0.0
        }
        
        # 1. 出界惩罚
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = env_bounds
        if (uav_position[0] < x_min or uav_position[0] > x_max or
            uav_position[1] < y_min or uav_position[1] > y_max):
            penalties['oob_penalty'] = -self.config.w_oob
        
        # 2. 停滞惩罚
        stagnation_penalty = self._calculate_stagnation_penalty(uav_position, current_time)
        penalties['stagnation_penalty'] = -stagnation_penalty
        
        # 3. 访问完用户后的特殊惩罚
        if self.all_users_visited:
            distance_to_end = np.linalg.norm(uav_position - end_position)
            
            # 时间浪费惩罚
            if self.all_users_visited_time is not None:
                time_wasted = current_time - self.all_users_visited_time
                if time_wasted > 5.0:  # 超过5秒没到终点
                    time_penalty = self.config.penalty_wasted_time * (time_wasted - 5.0)
                    penalties['time_waste_penalty'] = -time_penalty
            
            # 距离惩罚（距离终点过远的持续惩罚）
            if distance_to_end > self.config.end_position_tolerance * 2:
                distance_penalty = self.config.penalty_distance_from_end * (distance_to_end / 100.0)
                penalties['distance_penalty'] = -distance_penalty
        
        # 4. 未完成任务持续惩罚
        if not self.all_users_visited or not self._check_reached_end(uav_position, end_position):
            incomplete_penalty = self.config.penalty_incomplete_mission * 0.01  # 每步小惩罚
            penalties['incomplete_mission_penalty'] = -incomplete_penalty
        
        return penalties
    
    def _calculate_enhanced_terminal_rewards(self, reached_end: bool, current_time: float, 
                                           num_users: int, multipliers: Dict[str, float]) -> Dict[str, float]:
        """增强终端奖励"""
        rewards = {
            'terminal_reach_end': 0.0,
            'terminal_all_users': 0.0,
            'terminal_mission_complete': 0.0,
            'terminal_efficiency_bonus': 0.0
        }
        
        # 到达终点奖励
        if reached_end:
            rewards['terminal_reach_end'] = self.config.B_reach_end_base
        
        # 访问所有用户奖励
        if len(self.user_visited_flags) == num_users:
            rewards['terminal_all_users'] = self.config.B_all_users_visited
        
        # 完整任务奖励
        if (reached_end and 
            len(self.user_visited_flags) == num_users and
            200.0 <= current_time <= 300.0):
            completion_reward = self.config.B_mission_complete * multipliers['completion']
            rewards['terminal_mission_complete'] = completion_reward
            
            # 效率奖励（更快完成给予额外奖励）
            if self.all_users_visited_time is not None:
                completion_time = current_time - self.all_users_visited_time
                if completion_time < 30.0:  # 30秒内完成终点导航
                    efficiency_bonus = (30.0 - completion_time) * 200.0
                    rewards['terminal_efficiency_bonus'] = efficiency_bonus
                    print(f"⚡ 效率奖励！{completion_time:.1f}秒完成终点导航，获得{efficiency_bonus:.0f}奖励")
            
            print(f"🏆 任务完成！获得{completion_reward:.0f}完成奖励")
        
        return rewards
    
    def _calculate_stagnation_penalty(self, uav_position: np.ndarray, current_time: float) -> float:
        """计算停滞惩罚"""
        self.position_history.append(uav_position.copy())
        self.time_history.append(current_time)
        
        window_size = int(self.config.stagnation_time_window / self.config.time_step)
        if len(self.position_history) > window_size:
            self.position_history = self.position_history[-window_size:]
            self.time_history = self.time_history[-window_size:]
        
        if len(self.position_history) >= window_size:
            pos_variance = np.var(self.position_history, axis=0)
            total_variance = np.sum(pos_variance[:2])
            
            if total_variance < self.config.stagnation_threshold:
                # 如果访问完所有用户，停滞惩罚加重
                if self.all_users_visited:
                    return self.config.w_stagnation * 3.0
                else:
                    return self.config.w_stagnation
        
        return 0.0
    
    def _check_reached_end(self, uav_position: np.ndarray, end_position: np.ndarray) -> bool:
        """检查是否到达终点"""
        distance = np.linalg.norm(uav_position - end_position)
        return distance <= self.config.end_position_tolerance
    
    def _update_internal_state(self, uav_position: np.ndarray, user_positions: np.ndarray,
                              end_position: np.ndarray, current_time: float):
        """更新内部状态"""
        self.previous_position = uav_position.copy()
        
        # 更新访问完用户后的时间计数
        if self.all_users_visited and self.all_users_visited_time is not None:
            self.time_since_all_users_visited = current_time - self.all_users_visited_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return {
            'user_visit_times': dict(self.user_visit_times),
            'user_visited_flags': list(self.user_visited_flags),
            'user_visit_order': self.user_visit_order.copy(),
            'users_visited': len(self.user_visited_flags),
            'all_users_visited': self.all_users_visited,
            'end_guidance_activated': self.end_guidance_activated,
            'strong_guidance_activated': self.strong_guidance_activated,
            'time_since_all_users_visited': self.time_since_all_users_visited,
            'mission_completed': self.mission_completed
        }
