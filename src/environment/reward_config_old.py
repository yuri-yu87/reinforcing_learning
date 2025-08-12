"""
Reward Configuration for UAV Environment

This module defines reward parameters and configuration for the UAV environment.
Following the architectural principle: reward mechanism belongs to Environment layer.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RewardConfig:
    """
    Reward system configuration for UAV environment.
    
    Implements the optimized reward formula:
    r = w_rate · normalize(sum_rate) + w_goal · [F(d_end(s')) - F(d_end(s))] 
        + w_fair · Σ log(ε + service_i) - w_time · Δt
    """
    
    # Main reward weights
    w_rate: float = 0.8                    # Throughput weight (main objective)
    w_goal: float = 0.5                    # Goal weight (only active after all users visited)
    w_fair: float = 1.0                    # Strong fairness weight (force user rotation)
    w_time: float = 0.01                   # Minimal time penalty
    
    # Normalization parameters
    max_expected_throughput: float = 20.0  # For throughput normalization (higher to avoid early saturation)
    distance_normalization: float = 50.0   # Distance normalization parameter (d0)
    fairness_epsilon: float = 1e-6        # Small value to avoid log(0)
    
    # Mission completion tolerance
    end_position_tolerance: float = 5.0    # Meters tolerance for mission completion
    # Terminal bonus when reaching the end location the first time in an episode
    terminal_bonus: float = 800.0          # High bonus for mission completion with all users visited
    # Per-step penalty for stagnating (hovering/no displacement) when far from end
    hover_penalty: float = 0.1
    
    # 访问衰减机制 - 解决局部最优问题
    enable_visit_decay: bool = True         # 是否启用访问衰减
    visit_decay_radius: float = 20.0        # 用户访问半径(m)
    visit_decay_rate: float = 0.5           # 衰减倍数(0.5 = 50%衰减)
    visit_decay_threshold: float = 0.5      # 触发衰减的累积服务量
    
    # 用户访问完成门控机制 - 强制访问所有用户
    enable_visit_gating: bool = True
    min_visit_threshold: float = 2.0        # 更高阈值，确保真正靠近并服务过用户
    goal_reward_multiplier: float = 0.1     # 未访问完所有用户时的目标奖励倍数
    visited_goal_multiplier: float = 5.0    # 访问完所有用户后的目标奖励倍数
    
    # 用户专注机制 (改进版) - 距离+时间双重访问判定
    enable_user_focus: bool = True          # 启用用户专注机制
    
    # 新的访问完成判定条件：距离+时间双重要求
    visit_distance_threshold: float = 5.0  # 访问距离阈值：必须在此距离内
    visit_time_threshold: float = 5.0       # 访问时间阈值：必须在距离内停留此时间（秒）
    
    # 奖励权重
    focus_reward_multiplier: float = 1.0    # 当前专注用户的吞吐量奖励倍数
    non_focus_reward_multiplier: float = 0.0  # 非专注用户吞吐量奖励倍数（按要求设为0）
    visited_user_reward_multiplier: float = 0.0  # 已访问用户的吞吐量奖励倍数
    per_user_completion_bonus: float = 100.0     # 完成单个用户访问的一次性奖励（降低）
    
    # 距离增量奖励权重（类似w_rate）
    w_distance_approach: float = 1.0        # 距离接近奖励权重，与w_rate同等重要
    
    # 专注距离势函数参数
    w_focus: float = 0.4                    # 专注距离势函数权重
    focus_distance_normalization: float = 50.0  # 距离归一化参数 d0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'w_rate': self.w_rate,
            'w_goal': self.w_goal, 
            'w_fair': self.w_fair,
            'w_time': self.w_time,
            'max_expected_throughput': self.max_expected_throughput,
            'distance_normalization': self.distance_normalization,
            'fairness_epsilon': self.fairness_epsilon,
            'end_position_tolerance': self.end_position_tolerance,
            'terminal_bonus': self.terminal_bonus
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RewardConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def get_high_throughput_config(cls) -> 'RewardConfig':
        """Configuration optimized for high throughput."""
        return cls(
            w_rate=1.5,      # Higher throughput weight
            w_goal=0.2,      # Lower goal weight
            w_fair=0.1,      # Lower fairness weight
            w_time=0.01
        )
    
    @classmethod
    def get_balanced_config(cls) -> 'RewardConfig':
        """Balanced configuration for throughput + fairness."""
        return cls(
            w_rate=1.0,
            w_goal=0.3,
            w_fair=0.3,      # Higher fairness weight
            w_time=0.02
        )
    
    @classmethod
    def get_exploration_config(cls) -> 'RewardConfig':
        """Configuration for exploration and user coverage."""
        return cls(
            w_rate=0.8,
            w_goal=0.4,      # Higher goal weight
            w_fair=0.4,      # Higher fairness weight  
            w_time=0.05      # Higher time pressure
        )


class RewardCalculator:
    """
    Reward calculation logic for UAV environment.
    
    This class implements the actual reward computation based on the configuration.
    It maintains internal state for incremental potential function shaping.
    """
    
    def __init__(self, config: RewardConfig):
        """Initialize reward calculator with configuration."""
        self.config = config
        self.previous_distance_to_end = None
        self.user_cumulative_service = {}
        self._prev_fair_utility = 0.0
        # 访问衰减机制状态
        self.user_visit_decay = {}  # {user_id: decay_factor}
        self.has_reached_end = False
        # 用户专注与访问状态 (改进版)
        self.current_focus_user = None  # 当前专注的用户ID
        self.visited_users = set()       # 已访问完成的用户
        self.user_completion_given = set()  # 已发放一次性完成奖励的用户
        self.prev_focus_distance = None  # 上一步到专注用户的距离（用于势函数增量）
        
        # 新增：距离+时间访问状态跟踪
        self.user_close_time = {}        # 每个用户在访问距离内的累积时间
        self.all_users_visited = False   # 是否所有用户都已访问完成
    
    def reset(self, num_users: int):
        """Reset internal state for new episode."""
        self.previous_distance_to_end = None
        self.user_cumulative_service = {i: 0.0 for i in range(num_users)}
        self._prev_fair_utility = 0.0
        # 重置访问衰减状态
        self.user_visit_decay = {i: 1.0 for i in range(num_users)}  # 初始无衰减
        # 重置用户专注/访问状态 (改进版)
        self.current_focus_user = None
        self.visited_users = set()
        self.user_completion_given = set()
        self.prev_focus_distance = None
        
        # 重置距离+时间访问状态
        self.user_close_time = {i: 0.0 for i in range(num_users)}
        self.all_users_visited = False
        
        # 重置位置记录（用于移动检测）
        self.prev_uav_position = None
    
    def potential_function(self, distance_to_end: float) -> float:
        """Potential function: F(d) = 1/(1 + d/d0)"""
        return 1.0 / (1.0 + distance_to_end / self.config.distance_normalization)
    
    def focus_potential_function(self, distance_to_focus_user: float) -> float:
        """专注距离势函数: F(d) = 1/(1 + d/d0)"""
        return 1.0 / (1.0 + distance_to_focus_user / self.config.focus_distance_normalization)
    
    def check_all_users_visited(self) -> bool:
        """检查是否所有用户都已被充分访问"""
        if not self.config.enable_visit_gating:
            return True
        
        # 使用专注机制的 visited_users 集合来判断（更精确）
        if self.config.enable_user_focus:
            total_users = len(self.user_cumulative_service)
            return len(self.visited_users) >= total_users
        else:
            # 如果没有专注机制，回退到原来的服务阈值检查
            for user_id in self.user_cumulative_service:
                if self.user_cumulative_service[user_id] < self.config.min_visit_threshold:
                    return False
            return True
    
    def _get_user_distance(self, uav_position, user_id: int, user_positions) -> float:
        """计算UAV与指定用户的二维距离"""
        import numpy as np
        if user_positions is None or user_id >= len(user_positions):
            return float('inf')
        return float(np.linalg.norm(uav_position[:2] - user_positions[user_id][:2]))

    def _select_nearest_unvisited(self, uav_position, user_positions) -> int:
        best_id, best_dist = -1, float('inf')
        for uid in range(len(user_positions)):
            if uid in self.visited_users:
                continue
            d = self._get_user_distance(uav_position, uid, user_positions)
            if d < best_dist:
                best_dist, best_id = d, uid
        return best_id

    def _update_user_focus(self, uav_position, user_positions):
        """改进版专注机制：检查未访问用户列表，自动切换到终点导向模式"""
        if not self.config.enable_user_focus:
            return
        
        # 检查是否还有未访问用户
        unvisited_users = [uid for uid in range(len(user_positions)) if uid not in self.visited_users]
        
        if not unvisited_users:
            # 所有用户都已访问完成 - 切换到终点导向模式
            if not self.all_users_visited:
                print("🎯 所有用户访问完成！切换到终点导向模式")
                self.all_users_visited = True
                self.current_focus_user = None  # 清空专注用户
            return
        
        # 还有未访问用户 - 选择最近的未访问用户作为专注目标
        if self.current_focus_user is None or self.current_focus_user in self.visited_users:
            next_uid = self._select_nearest_unvisited(uav_position, user_positions)
            if next_uid != -1:
                old_focus = self.current_focus_user
                self.current_focus_user = next_uid
                self.prev_focus_distance = None  # 重置距离势函数
                print(f"专注切换: {old_focus} -> 用户{next_uid}")
    
    def get_user_focus_multipliers(self, num_users: int) -> Dict[int, float]:
        """获取每个用户的专注奖励倍数（专注=1.0，未专注=0.0，已访问=0.0）"""
        if not self.config.enable_user_focus:
            return {i: 1.0 for i in range(num_users)}

        multipliers: Dict[int, float] = {}
        for user_id in range(num_users):
            if hasattr(self, 'visited_users') and user_id in self.visited_users:
                multipliers[user_id] = self.config.visited_user_reward_multiplier  # 0.0
            elif user_id == self.current_focus_user:
                multipliers[user_id] = self.config.focus_reward_multiplier        # 1.0
            else:
                multipliers[user_id] = self.config.non_focus_reward_multiplier    # 0.0

        return multipliers
    
    def _update_visit_decay(self, uav_position, user_individual_throughputs):
        """更新用户访问衰减状态 - 个体化访问状态管理"""
        import numpy as np
        
        # 个体化访问状态管理：只对已充分服务的用户衰减
        for user_id, throughput in enumerate(user_individual_throughputs):
            if user_id in self.user_cumulative_service:
                # 如果该用户累积服务超过阈值，标记为"已访问"并衰减至接近0
                if self.user_cumulative_service[user_id] > self.config.visit_decay_threshold:
                    # 对已访问用户：衰减至接近0（几乎无奖励）
                    self.user_visit_decay[user_id] = 0.00  # 保留5%避免完全为0
                # 对未访问用户：保持满奖励
                else:
                    self.user_visit_decay[user_id] = 1.0
    
    def _check_user_visit_completion(self, uav_position, user_positions, time_step: float):
        """检查用户访问完成条件：距离+时间双重要求"""
        if not self.config.enable_user_focus or user_positions is None:
            return
        
        # 更新每个用户的在范围内时间
        for user_id in range(len(user_positions)):
            if user_id in self.visited_users:
                continue  # 跳过已访问完成的用户
                
            distance = self._get_user_distance(uav_position, user_id, user_positions)
            
            if distance <= self.config.visit_distance_threshold:
                # UAV在访问距离内，累积时间
                self.user_close_time[user_id] += time_step
                
                # 检查是否满足时间要求
                if self.user_close_time[user_id] >= self.config.visit_time_threshold:
                    if user_id not in self.visited_users:
                        print(f"✅ 用户{user_id}访问完成！距离={distance:.1f}m, 停留时间={self.user_close_time[user_id]:.1f}s")
                        self.visited_users.add(user_id)
                        return user_id  # 返回刚完成的用户ID
            else:
                # UAV离开访问范围，重置计时（可选：保持累积或重置）
                # 这里选择重置，要求连续停留
                if self.user_close_time[user_id] > 0:
                    self.user_close_time[user_id] = 0.0
        
        return None
    
    def calculate_reward(self, 
                        current_throughput: float,
                        uav_position,
                        end_position,
                        user_individual_throughputs,
                        user_positions=None,
                        time_step: float = 0.1) -> tuple[float, Dict[str, Any]]:
        """
        Calculate reward based on current state.
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        import numpy as np
        
        # 1. 改进版用户专注机制：距离+时间双重访问判定
        completion_bonus = 0.0
        distance_approach_reward = 0.0
        
        if self.config.enable_user_focus and user_positions is not None:
            # 首先检查用户访问完成条件（距离+时间）
            completed_user = self._check_user_visit_completion(uav_position, user_positions, time_step)
            
            if completed_user is not None:
                # 发放一次性完成奖励
                if completed_user not in self.user_completion_given:
                    completion_bonus = self.config.per_user_completion_bonus
                    self.user_completion_given.add(completed_user)
                    print(f"🎉 用户{completed_user}完成奖励发放: {completion_bonus:.1f}")
            
            # 更新专注用户选择（检查是否切换到终点导向模式）
            self._update_user_focus(uav_position, user_positions)
            
            # 计算距离接近奖励（类似w_rate的重要性）
            if self.current_focus_user is not None and not self.all_users_visited:
                current_focus_distance = self._get_user_distance(uav_position, self.current_focus_user, user_positions)
                
                if self.prev_focus_distance is not None:
                    # 距离增量奖励：F(d_prev) - F(d_curr)，靠近时为正奖励
                    F_prev = self.focus_potential_function(self.prev_focus_distance)
                    F_curr = self.focus_potential_function(current_focus_distance)
                    distance_approach_reward = self.config.w_distance_approach * (F_curr - F_prev)
                
                self.prev_focus_distance = current_focus_distance

        # 访问衰减：与位置无关
        if self.config.enable_visit_decay and len(user_individual_throughputs) > 0:
            self._update_visit_decay(uav_position, user_individual_throughputs)
        
        # 2. 吞吐量奖励 (条件性给予，避免hover陷阱)
        throughput_reward = 0.0
        normalized_throughput = 0.0
        
        # 只有在移动且靠近专注用户时才给予吞吐量奖励
        if not self.all_users_visited and self.current_focus_user is not None and user_positions is not None:
            focus_distance = self._get_user_distance(uav_position, self.current_focus_user, user_positions)
            
            # 检查是否移动（比较与上一步的距离差）
            moved_this_step = True
            if hasattr(self, 'prev_uav_position') and self.prev_uav_position is not None:
                displacement = np.linalg.norm(uav_position - self.prev_uav_position)
                moved_this_step = displacement > 1e-3  # 移动阈值1mm
            
            # 只有移动且在合理距离内才给予吞吐量奖励
            if moved_this_step and focus_distance < 30.0:
                if self.config.enable_user_focus and len(user_individual_throughputs) > 0:
                    focus_multipliers = self.get_user_focus_multipliers(len(user_individual_throughputs))
                    adjusted_throughput = 0.0
                    for user_id, user_throughput in enumerate(user_individual_throughputs):
                        focus_factor = focus_multipliers.get(user_id, 1.0)
                        adjusted_throughput += user_throughput * focus_factor
                    
                    normalized_throughput = np.clip(
                        adjusted_throughput / self.config.max_expected_throughput, 0.0, 1.0
                    )
                    # 移动奖励倍数：鼓励持续移动
                    movement_multiplier = min(displacement / 5.0, 1.0) if hasattr(self, 'prev_uav_position') and self.prev_uav_position is not None else 1.0
                    throughput_reward = self.config.w_rate * normalized_throughput * movement_multiplier
        
        # 记录当前位置供下次比较
        self.prev_uav_position = uav_position.copy()
        
        # 3. Mission progress reward (potential function increment) with visit gating
        current_distance = np.linalg.norm(uav_position - end_position)
        
        if self.previous_distance_to_end is not None:
            # Potential function shaping: γF(s') - F(s)
            F_current = self.potential_function(current_distance)
            F_previous = self.potential_function(self.previous_distance_to_end)
            potential_increment = F_current - F_previous
            
            # Apply visit gating: ZERO goal reward until all users visited
            if self.config.enable_visit_gating:
                all_visited = self.check_all_users_visited()
                if all_visited:
                    # 所有用户访问完成后，给予强烈的目标奖励
                    goal_reward = self.config.w_goal * self.config.visited_goal_multiplier * potential_increment
                else:
                    # 未访问完所有用户前，目标奖励完全为0
                    goal_reward = 0.0
            else:
                goal_reward = self.config.w_goal * potential_increment
        else:
            goal_reward = 0.0  # No increment for first step
        
        self.previous_distance_to_end = current_distance
        
        # 4. Fairness reward (终点导向模式下关闭)
        fair_reward = 0.0
        if not self.all_users_visited and len(user_individual_throughputs) > 0:
            # 只在用户访问阶段计算公平性奖励
            fair_utilities = [
                np.log(self.config.fairness_epsilon + service) 
                for service in self.user_cumulative_service.values()
            ]
            fair_utility_curr = float(np.sum(fair_utilities))
            fair_reward = self.config.w_fair * (fair_utility_curr - self._prev_fair_utility)
            self._prev_fair_utility = fair_utility_curr
        
        # 5. Time penalty (gentle)
        time_penalty = -self.config.w_time * time_step
        
        # Terminal bonus if reaching end for the first time (only if all users visited)
        reached_end = current_distance < self.config.end_position_tolerance
        if reached_end and self.config.enable_visit_gating:
            all_visited = self.check_all_users_visited()
            terminal_bonus = self.config.terminal_bonus if all_visited else 0.0
        elif reached_end:
            terminal_bonus = self.config.terminal_bonus
        else:
            terminal_bonus = 0.0

        # 6. 改进版奖励合成：双阶段设计
        if self.all_users_visited:
            # 终点导向阶段：只有目标奖励、时间惩罚和终端奖励
            total_reward = goal_reward + time_penalty + terminal_bonus
            print(f"🎯 终点导向模式：目标奖励={goal_reward:.3f}, 时间惩罚={time_penalty:.3f}")
        else:
            # 用户访问阶段：吞吐量、距离接近、完成奖励、公平性、时间惩罚
            goal_penalty = 0.0
            # 如果UAV在向终点移动，施加轻微目标惩罚
            if hasattr(self, 'previous_distance_to_end') and self.previous_distance_to_end is not None:
                if current_distance < self.previous_distance_to_end:  # Moving towards goal
                    goal_penalty = -2.0  # 轻微惩罚：鼓励先访问用户
            
            # 添加hover惩罚
            hover_penalty = 0.0
            if hasattr(self, 'prev_uav_position') and self.prev_uav_position is not None:
                displacement = np.linalg.norm(uav_position - self.prev_uav_position)
                if displacement < 1e-3:  # 基本没移动
                    hover_penalty = -self.config.hover_penalty
            
            total_reward = throughput_reward + distance_approach_reward + completion_bonus + fair_reward + time_penalty + goal_penalty + hover_penalty
        
        # Detailed breakdown for debugging/analysis
        reward_breakdown = {
            'throughput_reward': float(throughput_reward),
            'distance_approach_reward': float(distance_approach_reward),
            'completion_bonus': float(completion_bonus),
            'goal_reward': float(goal_reward),
            'fair_reward': float(fair_reward),
            'time_penalty': float(time_penalty),
            'hover_penalty': float(hover_penalty) if 'hover_penalty' in locals() else 0.0,
            'terminal_bonus': float(terminal_bonus),
            'total_reward': float(total_reward),
            'normalized_throughput': float(normalized_throughput),
            'distance_to_end': float(current_distance),
            'cumulative_services': dict(self.user_cumulative_service),
            'current_focus_user': int(self.current_focus_user) if (self.config.enable_user_focus and self.current_focus_user is not None) else None,
            'visited_users': list(self.visited_users) if hasattr(self, 'visited_users') else [],
            'focus_distance': float(self.prev_focus_distance) if self.prev_focus_distance is not None else None,
            'all_users_visited': bool(self.all_users_visited),
            'user_close_time': dict(self.user_close_time) if hasattr(self, 'user_close_time') else {},
            'uav_moved': bool(locals().get('moved_this_step', True)),
            'displacement': float(locals().get('displacement', 0.0))
        }
        
        return float(total_reward), reward_breakdown
