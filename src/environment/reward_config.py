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
    
    # 用户顺序访问机制 - 解决多用户吞吐量边界振荡
    enable_user_focus: bool = True          # 启用用户专注机制
    focus_threshold: float = 2.0            # 完成当前用户访问的阈值
    focus_reward_multiplier: float = 1.0    # 当前专注用户的吞吐量奖励倍数
    non_focus_reward_multiplier: float = 0.1  # 非专注用户的吞吐量奖励倍数（保留少量避免完全忽略）
    focus_stability_steps: int = 50         # 专注稳定步数：切换后需要保持专注的最小步数
    focus_switch_threshold: float = 2.0     # 吞吐量差异阈值：只有当新用户吞吐量显著更高时才切换
    # 最近用户访问与一次性奖励（新增）
    visited_user_reward_multiplier: float = 0.0  # 已访问用户的吞吐量奖励倍数
    focus_distance_requirement: float = 25.0     # 只有在该半径内才算“正在访问”
    focus_dwell_time_threshold: float = 3.0      # 判定完成访问所需的累计驻留时间(秒)
    per_user_completion_bonus: float = 200.0     # 完成单个用户访问的一次性奖励
    
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
        # 用户专注与访问状态
        self.current_focus_user = None  # 当前专注的用户ID
        self.user_completion_order = []  # 已完成访问的用户顺序
        self.focus_stability_counter = 0  # 当前专注的稳定步数计数器
        self.visited_users = set()       # 已访问完成的用户
        self.user_dwell_time = {}        # 每个用户累计驻留时间
        self.user_completion_given = set()  # 已发放一次性完成奖励的用户
    
    def reset(self, num_users: int):
        """Reset internal state for new episode."""
        self.previous_distance_to_end = None
        self.user_cumulative_service = {i: 0.0 for i in range(num_users)}
        self._prev_fair_utility = 0.0
        # 重置访问衰减状态
        self.user_visit_decay = {i: 1.0 for i in range(num_users)}  # 初始无衰减
        # 重置用户专注/访问状态
        self.current_focus_user = None
        self.user_completion_order = []
        self.focus_stability_counter = 0
        self.visited_users = set()
        self.user_dwell_time = {i: 0.0 for i in range(num_users)}
        self.user_completion_given = set()
    
    def potential_function(self, distance_to_end: float) -> float:
        """Potential function: F(d) = 1/(1 + d/d0)"""
        return 1.0 / (1.0 + distance_to_end / self.config.distance_normalization)
    
    def check_all_users_visited(self) -> bool:
        """检查是否所有用户都已被充分访问"""
        if not self.config.enable_visit_gating:
            return True
            
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
        """最近未访问优先。当前专注为空或已访问完成时重新选择。"""
        if not self.config.enable_user_focus:
            return
        if self.current_focus_user is None or self.current_focus_user in self.visited_users:
            next_uid = self._select_nearest_unvisited(uav_position, user_positions)
            if next_uid != -1:
                self.current_focus_user = next_uid
    
    def get_user_focus_multipliers(self, num_users: int) -> Dict[int, float]:
        """获取每个用户的专注奖励倍数"""
        if not self.config.enable_user_focus:
            return {i: 1.0 for i in range(num_users)}
        
        multipliers = {}
        for user_id in range(num_users):
            if user_id == self.current_focus_user:
                multipliers[user_id] = self.config.focus_reward_multiplier
            else:
                multipliers[user_id] = self.config.non_focus_reward_multiplier
        
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
                    self.user_visit_decay[user_id] = 0.05  # 保留5%避免完全为0
                # 对未访问用户：保持满奖励
                else:
                    self.user_visit_decay[user_id] = 1.0
    
    def calculate_reward(self, 
                        current_throughput: float,
                        uav_position,
                        end_position,
                        user_individual_throughputs,
                        time_step: float) -> tuple[float, Dict[str, Any]]:
        """
        Calculate reward based on current state.
        
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        import numpy as np
        
        # 1. Normalized throughput reward (main objective) with user focus and visit decay
        user_positions = None
        try:
            # 动态获取用户位置（避免循环依赖，运行时传入）
            from environment.uav_env import UAVEnvironment  # noqa: F401
        except Exception:
            pass
        
        # 我们依赖环境层把用户位置放到info里；此处无法直接获取，保守处理：仅按throughput计算
        # 但仍然实现驻留时间与一次性奖励的框架（需要环境传入用户位置时可生效）
        completion_bonus = 0.0
        if len(user_individual_throughputs) > 0 and hasattr(self, 'current_focus_user'):
            # 如果外层环境在info里附带用户位置，可在此注入；当前版本保持None安全行为
            if self.config.enable_user_focus:
                # 若能获取用户位置，则执行最近未访问逻辑；否则退化为仅按专注倍数加权
                try:
                    # 尝试从调用方注入（后续我们会让环境在breakdown中放入user_positions）
                    pass
                except Exception:
                    pass
        
        # 更新访问衰减状态（与位置无关，可正常使用）
        if self.config.enable_visit_decay and len(user_individual_throughputs) > 0:
            self._update_visit_decay(uav_position, user_individual_throughputs)
        
        # 应用用户专注机制：只专注当前目标用户的吞吐量
        if self.config.enable_user_focus and len(user_individual_throughputs) > 0:
            focus_multipliers = self.get_user_focus_multipliers(len(user_individual_throughputs))
            adjusted_throughput = 0.0
            for user_id, user_throughput in enumerate(user_individual_throughputs):
                focus_factor = focus_multipliers.get(user_id, 1.0)
                decay_factor = self.user_visit_decay.get(user_id, 1.0) if self.config.enable_visit_decay else 1.0
                adjusted_throughput += user_throughput * focus_factor * decay_factor
            
            normalized_throughput = np.clip(
                adjusted_throughput / self.config.max_expected_throughput, 0.0, 1.0
            )
        else:
            # 原有逻辑：应用个体化访问衰减机制
            normalized_throughput = np.clip(
                current_throughput / self.config.max_expected_throughput, 0.0, 1.0
            )
            if self.config.enable_visit_decay and len(user_individual_throughputs) > 0:
                adjusted_throughput = 0.0
                for user_id, user_throughput in enumerate(user_individual_throughputs):
                    decay_factor = self.user_visit_decay.get(user_id, 1.0)
                    adjusted_throughput += user_throughput * decay_factor
                
                normalized_throughput = np.clip(
                    adjusted_throughput / self.config.max_expected_throughput, 0.0, 1.0
                )
        
        throughput_reward = self.config.w_rate * normalized_throughput + completion_bonus
        
        # 2. Mission progress reward (potential function increment) with visit gating
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
        
        # 3. Fairness reward (encourage all users to receive service)
        # Use incremental potential: Δ(Σ log(ε + service_i)) to avoid ever-growing absolute bonus
        fair_reward = 0.0
        if len(user_individual_throughputs) > 0:
            # Fairness utility: Σ log(ε + service_i) (累积服务已在上面更新)
            fair_utilities = [
                np.log(self.config.fairness_epsilon + service) 
                for service in self.user_cumulative_service.values()
            ]
            fair_utility_curr = float(np.sum(fair_utilities))
            fair_reward = self.config.w_fair * (fair_utility_curr - self._prev_fair_utility)
            self._prev_fair_utility = fair_utility_curr
        
        # 4. Time penalty (gentle)
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

        # CRITICAL: Precise user visit gating - block only goal rewards, keep throughput
        if self.config.enable_visit_gating and not self.check_all_users_visited():
            # 未访问完所有用户前：保留吞吐量奖励（距离相关），阻断目标奖励
            goal_penalty = 0.0
            # 如果UAV在向终点移动，施加目标惩罚
            if hasattr(self, 'previous_distance_to_end') and self.previous_distance_to_end is not None:
                if current_distance < self.previous_distance_to_end:  # Moving towards goal
                    goal_penalty = -8.0  # 强惩罚：未访问完前靠近终点
            
            total_reward = throughput_reward + fair_reward + time_penalty + goal_penalty
            # 目标奖励和终端奖励完全阻断，直到访问完所有用户
        else:
            # 访问完所有用户后：完整奖励机制
            total_reward = goal_reward + time_penalty + terminal_bonus
        
        # Detailed breakdown for debugging/analysis
        reward_breakdown = {
            'throughput_reward': float(throughput_reward),
            'goal_reward': float(goal_reward),
            'fair_reward': float(fair_reward),
            'time_penalty': float(time_penalty),
            'total_reward': float(total_reward),
            'normalized_throughput': float(normalized_throughput),
            'distance_to_end': float(current_distance),
            'cumulative_services': dict(self.user_cumulative_service),
            'current_focus_user': int(self.current_focus_user) if (self.config.enable_user_focus and self.current_focus_user is not None) else None,
            'user_completion_order': list(self.user_completion_order) if self.config.enable_user_focus else None,
            'terminal_bonus': float(terminal_bonus)
        }
        
        return float(total_reward), reward_breakdown
