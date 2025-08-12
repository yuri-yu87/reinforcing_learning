"""
激进增量奖励机制设计
完全移除绝对位置奖励，只通过距离增量和任务完成驱动行为
强迫UAV移动，防止任何形式的悬停利用
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class RewardConfig:
    """激进增量奖励配置 - 纯增量驱动 + 大奖励完成激励"""
    
    # 核心奖励权重 (训练策略优化 - 平衡引导与任务)
    w_approach: float = 1.0              # 距离增量奖励权重（恢复温和引导）
    w_completion: float = 5000.0         # 任务完成奖励权重（大幅提升！）
    w_time: float = 0.01                 # 时间惩罚权重（进一步降低）
    w_movement_bonus: float = 0.05       # 移动奖励（恢复基础激励）
    w_incomplete_penalty: float = 2.0    # 未完成用户惩罚（适中）
    w_position_guidance: float = 0.5     # 位置引导奖励（降低，避免与距离奖励冲突）
    w_constraint_violation: float = 100.0   # 约束违反惩罚（大幅降低，允许学习）
    
    # 访问判定条件 (距离+时间双重条件)
    visit_distance_threshold: float = 5.0   # 访问距离阈值
    visit_time_threshold: float = 3.0        # 访问时间阈值（保留，但不使用）
    max_service_time: float = 20.0           # 最大服务时长（防止悬停exploit）
    
    # 环境参数
    end_position_tolerance: float = 8.0     # 终点容忍范围
    terminal_bonus: float = 3000.0          # 终点到达奖励（极大提升！10倍于exploit收益）
    
    # 约束强化学习参数
    min_approach_distance: float = 0.5      # 最小接近距离
    approach_reward_cap: float = 2.0        # 增量奖励上限
    max_reward_distance: float = 80.0       # 奖励范围（恢复合理范围）
    hover_penalty: float = 2.0              # 悬停惩罚（温和）
    min_movement_for_reward: float = 0.5    # 移动奖励最小距离
    position_guidance_range: float = 80.0   # 位置引导范围（恢复）
    
    # 温和约束参数（训练友好）
    min_movement_per_window: float = 2.0    # 每个时间窗口最小移动距离（大幅降低）
    movement_check_window: int = 200        # 移动检查窗口（步数）（延长）
    min_progress_rate: float = 0.02         # 最小进度率（大幅降低）
    progress_check_interval: int = 300      # 进度检查间隔（步数）（大幅延长）
    max_stagnation_steps: int = 500         # 最大停滞步数（大幅放宽）
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于环境信息收集）"""
        return {
            'w_approach': self.w_approach,
            'w_completion': self.w_completion,
            'w_time': self.w_time,
            'w_movement_bonus': self.w_movement_bonus,
            'w_incomplete_penalty': self.w_incomplete_penalty,
            'w_position_guidance': self.w_position_guidance,
            'w_constraint_violation': self.w_constraint_violation,
                    'visit_distance_threshold': self.visit_distance_threshold,
        'visit_time_threshold': self.visit_time_threshold,
        'max_service_time': self.max_service_time,
            'end_position_tolerance': self.end_position_tolerance,
            'terminal_bonus': self.terminal_bonus,
            'min_approach_distance': self.min_approach_distance,
            'approach_reward_cap': self.approach_reward_cap,
            'max_reward_distance': self.max_reward_distance,
            'hover_penalty': self.hover_penalty,
            'min_movement_for_reward': self.min_movement_for_reward,
            'position_guidance_range': self.position_guidance_range
        }


class RewardCalculator:
    """
    激进增量奖励计算器
    
    核心理念：
    1. 完全移除绝对位置奖励 - 断绝悬停获利
    2. 严格距离增量奖励 - 只有真正接近才给奖励
    3. 移动奖励 - 鼓励任何形式的移动
    4. 大幅任务完成奖励 - 强化正确目标
    5. 强化时间压力 - 避免无效徘徊
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.reset(2)
    
    def reset(self, num_users: int):
        """重置状态"""
        self.visited_users = set()
        self.user_entered = set()  # 新增：跟踪已进入用户区域的状态
        self.user_close_time = {i: 0.0 for i in range(num_users)}
        self.user_service_time = {i: 0.0 for i in range(num_users)}  # 服务时长跟踪
        
        # 约束检查机制
        self.position_history = []  # 位置历史
        self.movement_window = []   # 移动窗口
        self.progress_check_steps = 0  # 进度检查步数
        self.last_progress_distance = None  # 上次进度检查时的距离
        self.stagnation_steps = 0   # 停滞步数计数
        self.current_focus_user = 0  # 从用户0开始
        self.all_users_visited = False
        self.user_completion_given = set()
        
        # 距离跟踪状态（用于增量奖励）
        self.prev_focus_distance = None
        self.prev_goal_distance = None
    

    
    def _check_out_of_bounds(self, uav_position) -> bool:
        """检查UAV是否越界（简单边界检查）"""
        # 假设飞行区域为 [0, 100] x [0, 100] x [0, 100]
        return (uav_position[0] < 0 or uav_position[0] > 100 or
                uav_position[1] < 0 or uav_position[1] > 100 or
                uav_position[2] < 0 or uav_position[2] > 100)
    
    def _get_user_distance(self, uav_position, user_id: int, user_positions) -> float:
        """计算UAV与用户的距离"""
        if user_positions is None or user_id >= len(user_positions):
            return float('inf')
        return float(np.linalg.norm(uav_position[:2] - user_positions[user_id][:2]))

    def _check_user_completion(self, uav_position, user_positions, time_step: float):
        """检查用户访问完成（进入+离开双重判定+服务时长限制）- 只检查当前专注用户"""
        # 清理已访问用户的进入状态和服务时长
        for user_id in list(self.user_entered):
            if user_id in self.visited_users:
                self.user_entered.remove(user_id)
                self.user_service_time[user_id] = 0.0  # 重置服务时长
        
        # 只检查当前专注的用户（与观察状态保持一致）
        if self.current_focus_user is None or self.current_focus_user >= len(user_positions):
            return None
            
        user_id = self.current_focus_user
        
        # 如果当前专注用户已经被访问，跳过
        if user_id in self.visited_users:
            return None
            
        distance = self._get_user_distance(uav_position, user_id, user_positions)
        
        # 检查进入状态
        if distance <= self.config.visit_distance_threshold:
            if user_id not in self.user_entered:
                self.user_entered.add(user_id)
                self.user_service_time[user_id] = 0.0  # 开始计时
                print(f"🎯 UAV进入用户{user_id}服务区域，距离={distance:.1f}m")
            else:
                # 已进入，累计服务时长
                self.user_service_time[user_id] += time_step
                
                # 检查是否超过最大服务时长（防止悬停exploit）
                if self.user_service_time[user_id] >= self.config.max_service_time:
                    self.visited_users.add(user_id)
                    self.user_entered.remove(user_id)
                    self.user_service_time[user_id] = 0.0
                    print(f"⏰ 用户{user_id}服务时长达到上限！自动完成访问")
                    return user_id
        
        # 检查离开状态（已进入且现在离开）
        elif user_id in self.user_entered and distance > self.config.visit_distance_threshold:
            self.visited_users.add(user_id)
            self.user_entered.remove(user_id)  # 立即清理进入状态
            self.user_service_time[user_id] = 0.0  # 重置服务时长
            print(f"✅ 用户{user_id}访问完成！(进入→离开)")
            return user_id
        
        return None
    
    def _update_focus(self, uav_position, user_positions):
        """更新专注用户 - 稳定专注策略，避免频繁切换"""
        # 检查是否所有用户都访问完成
        if len(self.visited_users) >= len(user_positions):
            self.all_users_visited = True
            self.current_focus_user = None
            return
        
        # 如果当前专注用户已被访问，或者还没有专注用户，则选择新的
        if (self.current_focus_user is None or 
            self.current_focus_user in self.visited_users):
            
            # 选择第一个未访问用户（稳定策略，避免频繁切换）
            for user_id in range(len(user_positions)):
                if user_id not in self.visited_users:
                    if user_id != self.current_focus_user:  # 只有真正切换时才重置
                        self.current_focus_user = user_id
                        # 重置距离跟踪状态（防止增量奖励基于错误的基线）
                        self.prev_focus_distance = None
                        # 重置新专注用户的服务时长
                        self.user_service_time[user_id] = 0.0
                        print(f"🎯 专注切换到用户{user_id}，重置距离跟踪")
                    break
    
    def calculate_reward(self, uav_position, end_position, user_positions, 
                        user_individual_throughputs=None, prev_position=None, time_step=0.1) -> tuple[float, Dict[str, Any]]:
        """
        激进增量奖励计算
        
        Args:
            uav_position: UAV当前位置
            end_position: 终点位置
            user_positions: 用户位置列表
            user_individual_throughputs: 不使用
            prev_position: UAV上一步位置（用于移动检测）
            time_step: 时间步长
            
        返回: (总奖励, 奖励详情)
        """
        
        # 1. 检查用户访问完成（大奖励）
        completion_bonus = 0.0
        completed_user = self._check_user_completion(uav_position, user_positions, time_step)
        if completed_user is not None and completed_user not in self.user_completion_given:
            completion_bonus = self.config.w_completion
            self.user_completion_given.add(completed_user)
            print(f"🎉 用户{completed_user}访问完成！奖励={completion_bonus}")
        
        # 2. 更新专注状态
        self._update_focus(uav_position, user_positions)
        
        # 3. 计算距离增量奖励（唯一持续奖励）
        # 恢复距离增量奖励（温和引导）
        if self.all_users_visited:
            approach_reward = self._calculate_goal_approach_reward(uav_position, end_position)
            target_info = "Goal"
        else:
            approach_reward = self._calculate_user_approach_reward(uav_position, user_positions)
            target_info = f"User{self.current_focus_user}"
        
        # 4. 移动奖励（恢复基础激励）
        movement_reward = 0.0
        if prev_position is not None:
            displacement = np.linalg.norm(uav_position - prev_position)
            if displacement >= self.config.min_movement_for_reward:
                movement_reward = self.config.w_movement_bonus * displacement
        
        # 5. 未完成用户惩罚（温和鼓励访问所有用户）
        incomplete_penalty = 0.0
        if not self.all_users_visited and user_positions is not None:
            num_unvisited = len(user_positions) - len(self.visited_users)
            incomplete_penalty = -self.config.w_incomplete_penalty * num_unvisited
        
        # 6. 位置引导奖励（温和引导到目标）
        position_guidance = 0.0
        if self.all_users_visited:
            # 朝向终点的引导
            distance_to_end = np.linalg.norm(uav_position - end_position)
            if distance_to_end <= self.config.position_guidance_range:
                position_guidance = self.config.w_position_guidance * (1.0 - distance_to_end / self.config.position_guidance_range)
        else:
            # 朝向当前专注用户的引导（只有未访问的用户）
            if (self.current_focus_user is not None and 
                self.current_focus_user < len(user_positions) and 
                self.current_focus_user not in self.visited_users):  # 关键修复：只引导未访问用户
                
                focus_user_position = user_positions[self.current_focus_user]
                distance_to_user = np.linalg.norm(uav_position - focus_user_position)
                if distance_to_user <= self.config.position_guidance_range:
                    position_guidance = self.config.w_position_guidance * (1.0 - distance_to_user / self.config.position_guidance_range)
        
        # 7. 约束违反检查（硬约束防止exploit）
        constraint_violation_penalty = self._check_constraints(uav_position, user_positions, time_step)
        
        # 8. 智能悬停惩罚（温和阻止悬停）
        hover_penalty = 0.0
        if prev_position is not None:
            displacement = np.linalg.norm(uav_position - prev_position)
            if displacement < 0.5:  # 几乎没有移动
                # 检查是否在任何用户的服务区域内
                in_service_area = False
                if not self.all_users_visited and user_positions is not None:
                    for user_id in range(len(user_positions)):
                        if user_id not in self.visited_users:
                            distance = self._get_user_distance(uav_position, user_id, user_positions)
                            if distance <= self.config.visit_distance_threshold * 1.2:  # 稍大范围
                                in_service_area = True
                                break
                
                # 只在非服务区域惩罚悬停
                if not in_service_area:
                    hover_penalty = -self.config.hover_penalty
        
        # 8. 时间惩罚（温和紧迫感）
        time_penalty = -self.config.w_time * time_step
        
        # 9. 终端奖励（大幅提升）
        terminal_bonus = 0.0
        if self.all_users_visited:
            distance_to_end = np.linalg.norm(uav_position - end_position)
            if distance_to_end < self.config.end_position_tolerance:
                terminal_bonus = self.config.terminal_bonus
                print(f"Goal reached! Reward={terminal_bonus}")
        
        # 10. 总奖励计算（约束强化学习）
        total_reward = (approach_reward + movement_reward + completion_bonus + 
                       incomplete_penalty + position_guidance + constraint_violation_penalty + hover_penalty + time_penalty + terminal_bonus)
        
        # 11. 奖励详情
        breakdown = {
            'approach_reward': float(approach_reward),  # 已移除
            'movement_reward': float(movement_reward),  # 已移除
            'completion_bonus': float(completion_bonus),
            'incomplete_penalty': float(incomplete_penalty),
            'position_guidance': float(position_guidance),
            'constraint_violation_penalty': float(constraint_violation_penalty),
            'hover_penalty': float(hover_penalty),
            'time_penalty': float(time_penalty),
            'terminal_bonus': float(terminal_bonus),
            'total_reward': float(total_reward),
            'target': target_info,
            'current_focus_user': self.current_focus_user,
            'visited_users': list(self.visited_users),
            'all_users_visited': self.all_users_visited,
            'user_close_time': dict(self.user_close_time),
            'prev_focus_distance': self.prev_focus_distance,
            'prev_goal_distance': self.prev_goal_distance
        }
        
        return float(total_reward), breakdown
    
    def _check_constraints(self, uav_position, user_positions, time_step):
        """检查约束违反 - 硬约束防止exploit行为"""
        total_penalty = 0.0
        
        # 记录当前位置
        self.position_history.append(uav_position.copy())
        
        # 计算当前移动距离
        current_movement = 0.0
        if len(self.position_history) > 1:
            current_movement = np.linalg.norm(uav_position - self.position_history[-2])
        
        # 维护移动窗口
        self.movement_window.append(current_movement)
        if len(self.movement_window) > self.config.movement_check_window:
            self.movement_window.pop(0)
        
        # 约束1: 移动约束 - 必须保持最低移动量
        if len(self.movement_window) >= self.config.movement_check_window:
            total_movement = sum(self.movement_window)
            if total_movement < self.config.min_movement_per_window:
                penalty = -self.config.w_constraint_violation * (1 + (self.config.min_movement_per_window - total_movement) / self.config.min_movement_per_window)
                total_penalty += penalty
                print(f"⚠️ 移动约束违反！窗口移动={total_movement:.1f}, 要求>{self.config.min_movement_per_window}, 惩罚={penalty:.1f}")
        
        # 约束2: 进度约束 - 必须朝目标前进（温和版本）
        self.progress_check_steps += 1
        if self.progress_check_steps >= self.config.progress_check_interval:
            current_target_distance = self._get_current_target_distance(uav_position, user_positions)
            
            if self.last_progress_distance is not None:
                progress = self.last_progress_distance - current_target_distance
                required_progress = self.config.min_progress_rate * self.config.progress_check_interval * time_step
                
                # 严格进度要求：必须朝目标前进
                if progress < required_progress:  # 不允许倒退或停滞
                    penalty_factor = min(3.0, abs(progress - required_progress) / required_progress)  # 提高惩罚倍数
                    penalty = -self.config.w_constraint_violation * penalty_factor
                    total_penalty += penalty
                    print(f"⚠️ 进度约束违反！实际进度={progress:.1f}, 要求>{required_progress:.1f}, 惩罚={penalty:.1f}")
            
            # 重置进度检查
            self.last_progress_distance = current_target_distance
            self.progress_check_steps = 0
        
        # 约束3: 停滞约束 - 不能长时间无移动
        if current_movement < 0.1:  # 几乎没有移动
            self.stagnation_steps += 1
            if self.stagnation_steps > self.config.max_stagnation_steps:
                penalty = -self.config.w_constraint_violation * (1 + (self.stagnation_steps - self.config.max_stagnation_steps) / 100.0)
                total_penalty += penalty
                print(f"⚠️ 停滞约束违反！停滞步数={self.stagnation_steps}, 最大={self.config.max_stagnation_steps}, 惩罚={penalty:.1f}")
        else:
            self.stagnation_steps = 0  # 重置停滞计数
        
        return total_penalty
    
    def _get_current_target_distance(self, uav_position, user_positions):
        """获取到当前目标的距离"""
        if self.all_users_visited:
            # 朝向终点
            return np.linalg.norm(uav_position - np.array([80, 80, 50]))
        elif self.current_focus_user is not None and self.current_focus_user < len(user_positions):
            # 朝向当前专注用户
            return self._get_user_distance(uav_position, self.current_focus_user, user_positions)
        else:
            # 朝向最近的未访问用户
            min_distance = float('inf')
            for user_id in range(len(user_positions)):
                if user_id not in self.visited_users:
                    distance = self._get_user_distance(uav_position, user_id, user_positions)
                    min_distance = min(min_distance, distance)
            return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_user_approach_reward(self, uav_position, user_positions) -> float:
        """计算用户阶段的严格距离增量奖励"""
        if self.current_focus_user is None or self.current_focus_user >= len(user_positions):
            return 0.0
        
        focus_user_position = user_positions[self.current_focus_user]
        current_distance = np.linalg.norm(uav_position - focus_user_position)
        
        # 严格的距离增量奖励
        approach_reward = 0.0
        if self.prev_focus_distance is not None:
            distance_improvement = self.prev_focus_distance - current_distance
            
            # 严格条件：显著接近 + 合理距离范围 + 不能太远
            if (distance_improvement >= self.config.min_approach_distance and 
                current_distance <= self.config.max_reward_distance and
                current_distance > 5.0):  # 不能太近（防止在目标周围振荡）
                
                approach_reward = min(
                    self.config.w_approach * distance_improvement,
                    self.config.approach_reward_cap
                )
        
        # 更新距离记录
        self.prev_focus_distance = current_distance
        
        # 调试输出（每50步显示一次）
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        if self._debug_counter % 50 == 0:
            print(f"🔍 用户{self.current_focus_user}: 距离={current_distance:.1f}m, "
                  f"增量奖励={approach_reward:.3f}")
        
        return approach_reward
    
    def _calculate_goal_approach_reward(self, uav_position, end_position) -> float:
        """计算终点阶段的严格距离增量奖励"""
        current_distance = np.linalg.norm(uav_position - end_position)
        
        # 严格的距离增量奖励
        approach_reward = 0.0
        if self.prev_goal_distance is not None:
            distance_improvement = self.prev_goal_distance - current_distance
            
            # 严格条件：显著接近 + 合理距离范围
            if (distance_improvement >= self.config.min_approach_distance and 
                current_distance <= self.config.max_reward_distance and
                current_distance > 5.0):
                
                approach_reward = min(
                    self.config.w_approach * distance_improvement,
                    self.config.approach_reward_cap
                )
        
        # 更新距离记录
        self.prev_goal_distance = current_distance
        
        return approach_reward

# 测试激进增量奖励机制
def test_radical_approach_reward():
    config = RewardConfig()
    calculator = RewardCalculator(config)
    
    # 测试场景
    uav_pos = np.array([0.0, 0.0, 50.0])
    end_pos = np.array([80.0, 80.0, 50.0])
    user_positions = np.array([[15.0, 75.0, 0.0], [75.0, 15.0, 0.0]])
    
    print("=== 平衡增量奖励机制测试 ===")
    
    # 测试1：静止状态（应该有温和惩罚和位置引导）
    print("\n--- 测试1: 静止状态 ---")
    reward1, breakdown1 = calculator.calculate_reward(
        uav_pos, end_pos, user_positions
    )
    print(f"静止奖励: {reward1:.4f}")
    print(f"增量奖励: {breakdown1['approach_reward']:.4f}")
    print(f"移动奖励: {breakdown1['movement_reward']:.4f}")
    print(f"未完成惩罚: {breakdown1['incomplete_penalty']:.4f}")
    print(f"位置引导: {breakdown1['position_guidance']:.4f}")
    print(f"时间惩罚: {breakdown1['time_penalty']:.4f}")
    print(f"总奖励: {breakdown1['total_reward']:.4f}")
    
    # 测试2：大幅接近用户（应该有显著奖励）
    print("\n--- 测试2: 大幅接近用户0 ---")
    new_pos = np.array([10.0, 60.0, 50.0])  # 大幅向用户0移动
    reward2, breakdown2 = calculator.calculate_reward(
        new_pos, end_pos, user_positions, prev_position=uav_pos
    )
    print(f"大幅接近奖励: {reward2:.4f}")
    print(f"增量奖励: {breakdown2['approach_reward']:.4f}")
    print(f"移动奖励: {breakdown2['movement_reward']:.4f}")
    print(f"总奖励: {breakdown2['total_reward']:.4f}")
    
    # 测试3：微小移动（应该被拒绝增量奖励）
    print("\n--- 测试3: 微小移动 ---")
    tiny_pos = np.array([10.5, 60.5, 50.0])  # 微小移动
    reward3, breakdown3 = calculator.calculate_reward(
        tiny_pos, end_pos, user_positions, prev_position=new_pos
    )
    print(f"微小移动奖励: {reward3:.4f}")
    print(f"增量奖励: {breakdown3['approach_reward']:.4f}")
    print(f"移动奖励: {breakdown3['movement_reward']:.4f}")
    print(f"总奖励: {breakdown3['total_reward']:.4f}")
    
    # 测试4：用户访问完成
    print("\n--- 测试4: 用户访问完成 ---")
    calculator.visited_users.add(0)
    calculator.user_completion_given.add(0)
    reward4, breakdown4 = calculator.calculate_reward(
        np.array([15.0, 75.0, 50.0]), end_pos, user_positions
    )
    print(f"用户完成奖励: {reward4:.4f}")
    print(f"完成奖励: {breakdown4['completion_bonus']:.4f}")
    
    # 测试5：悬停惩罚
    print("\n--- 测试5: 悬停惩罚 ---")
    hover_pos = np.array([15.0, 75.0, 50.0])  # 悬停在同一位置
    reward5, breakdown5 = calculator.calculate_reward(
        hover_pos, end_pos, user_positions, prev_position=hover_pos
    )
    print(f"悬停奖励: {reward5:.4f}")
    print(f"悬停惩罚: {breakdown5['hover_penalty']:.4f}")
    print(f"时间惩罚: {breakdown5['time_penalty']:.4f}")
    print(f"总奖励: {breakdown5['total_reward']:.4f}")
    
    # 测试6：终点到达
    print("\n--- 测试6: 终点到达 ---")
    calculator.all_users_visited = True
    reward6, breakdown6 = calculator.calculate_reward(
        np.array([80.0, 80.0, 50.0]), end_pos, user_positions
    )
    print(f"终点奖励: {reward6:.4f}")
    print(f"终端奖励: {breakdown6['terminal_bonus']:.4f}")
    print(f"总奖励: {breakdown6['total_reward']:.4f}")
    
    return calculator


# 测试距离引导奖励机制
def test_distance_approach_reward():
    config = RewardConfig()
    calculator = RewardCalculator(config)
    
    # 测试场景
    uav_pos = np.array([0.0, 0.0, 50.0])
    end_pos = np.array([80.0, 80.0, 50.0])
    user_positions = np.array([[15.0, 75.0, 0.0], [75.0, 15.0, 0.0]])
    
    print("=== 距离引导奖励机制测试 ===")
    
    # 模拟UAV向用户0移动的过程
    print("\n--- 模拟UAV向用户0接近过程 ---")
    
    prev_pos = uav_pos
    for i in range(10):
        # 模拟UAV向用户0移动
        target_user = user_positions[0]
        direction = (target_user - uav_pos)
        direction = direction / np.linalg.norm(direction)  # 单位方向向量
        current_pos = prev_pos + direction * 3.0  # 每步移动3米
        
        reward, breakdown = calculator.calculate_reward(
            current_pos, end_pos, user_positions, None, prev_pos
        )
        
        distance_to_target = np.linalg.norm(current_pos - target_user)
        print(f"步骤{i+1}: 距离用户0={distance_to_target:.1f}m, 奖励={reward:.3f}")
        prev_pos = current_pos
        
        # 如果已经很接近用户0，停止测试
        if distance_to_target < 5.0:
            break
    
    print("\n=== 测试完成 ===")
    return True


if __name__ == "__main__":
    test_radical_approach_reward()
