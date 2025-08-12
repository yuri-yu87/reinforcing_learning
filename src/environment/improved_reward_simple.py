"""
简化版奖励机制设计
解决奖励冗余和UAV hover问题
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

@dataclass
class SimpleRewardConfig:
    """简化版奖励配置 - 消除冗余，专注核心目标"""
    
    # 核心奖励权重 (只保留最重要的)
    w_approach: float = 1.0          # 接近奖励权重（核心驱动力）
    w_goal: float = 2.0              # 目标奖励权重（终点导向）
    w_time: float = 0.02             # 时间惩罚（防止拖延）
    
    # 访问判定条件 (你的改进：距离+时间)
    visit_distance_threshold: float = 15.0   # 访问距离阈值
    visit_time_threshold: float = 3.0        # 访问时间阈值
    per_user_completion_bonus: float = 30.0  # 完成奖励（降低）
    
    # 环境参数
    distance_normalization: float = 50.0
    end_position_tolerance: float = 8.0
    terminal_bonus: float = 200.0
    
    # 移动激励参数
    movement_bonus: float = 0.1      # 移动奖励（鼓励探索）
    hover_penalty: float = 2.0       # 悬停惩罚（强化）


class SimpleRewardCalculator:
    """
    简化版奖励计算器
    
    核心理念：
    1. 用户访问阶段：只有距离接近奖励 + 移动奖励 + 完成奖励
    2. 终点导向阶段：只有目标接近奖励 + 移动奖励  
    3. 完全去除吞吐量奖励（避免hover陷阱）
    """
    
    def __init__(self, config: SimpleRewardConfig):
        self.config = config
        self.reset(2)
    
    def reset(self, num_users: int):
        """重置状态"""
        self.visited_users = set()
        self.user_close_time = {i: 0.0 for i in range(num_users)}
        self.current_focus_user = 0  # 从用户0开始
        self.all_users_visited = False
        self.prev_focus_distance = None
        self.prev_goal_distance = None
        self.user_completion_given = set()
    
    def potential_function(self, distance: float) -> float:
        """势函数: F(d) = 1/(1 + d/d0)"""
        return 1.0 / (1.0 + distance / self.config.distance_normalization)
    
    def _get_user_distance(self, uav_position, user_id: int, user_positions) -> float:
        """计算UAV与用户的距离"""
        if user_positions is None or user_id >= len(user_positions):
            return float('inf')
        return float(np.linalg.norm(uav_position[:2] - user_positions[user_id][:2]))
    
    def _check_user_completion(self, uav_position, user_positions, time_step: float):
        """检查用户访问完成（距离+时间双重判定）"""
        for user_id in range(len(user_positions)):
            if user_id in self.visited_users:
                continue
            
            distance = self._get_user_distance(uav_position, user_id, user_positions)
            
            if distance <= self.config.visit_distance_threshold:
                self.user_close_time[user_id] += time_step
                if self.user_close_time[user_id] >= self.config.visit_time_threshold:
                    if user_id not in self.visited_users:
                        self.visited_users.add(user_id)
                        return user_id
            else:
                self.user_close_time[user_id] = 0.0  # 重置计时
        
        return None
    
    def _update_focus(self, uav_position, user_positions):
        """更新专注用户"""
        # 检查是否所有用户都访问完成
        if len(self.visited_users) >= len(user_positions):
            self.all_users_visited = True
            self.current_focus_user = None
            return
        
        # 选择最近的未访问用户
        best_user, best_distance = None, float('inf')
        for user_id in range(len(user_positions)):
            if user_id not in self.visited_users:
                distance = self._get_user_distance(uav_position, user_id, user_positions)
                if distance < best_distance:
                    best_distance = distance
                    best_user = user_id
        
        if best_user != self.current_focus_user:
            self.current_focus_user = best_user
            self.prev_focus_distance = None  # 重置距离
    
    def calculate_reward(self, uav_position, end_position, user_positions, 
                        prev_position=None, time_step=0.1) -> tuple[float, Dict[str, Any]]:
        """
        简化版奖励计算
        
        返回: (总奖励, 奖励详情)
        """
        
        # 1. 检查用户访问完成
        completion_bonus = 0.0
        completed_user = self._check_user_completion(uav_position, user_positions, time_step)
        if completed_user is not None and completed_user not in self.user_completion_given:
            completion_bonus = self.config.per_user_completion_bonus
            self.user_completion_given.add(completed_user)
            print(f"✅ 用户{completed_user}访问完成！奖励={completion_bonus}")
        
        # 2. 更新专注状态
        self._update_focus(uav_position, user_positions)
        
        # 3. 核心奖励计算
        if self.all_users_visited:
            # 终点导向阶段：只关注到达终点
            approach_reward = self._calculate_goal_approach_reward(uav_position, end_position)
            target_info = "Goal"
        else:
            # 用户访问阶段：只关注接近当前专注用户
            approach_reward = self._calculate_user_approach_reward(uav_position, user_positions)
            target_info = f"User{self.current_focus_user}"
        
        # 4. 移动奖励（鼓励探索，避免hover）
        movement_reward = self._calculate_movement_reward(uav_position, prev_position)
        
        # 5. 时间惩罚
        time_penalty = -self.config.w_time * time_step
        
        # 6. 终端奖励
        terminal_bonus = 0.0
        if self.all_users_visited:
            distance_to_end = np.linalg.norm(uav_position - end_position)
            if distance_to_end < self.config.end_position_tolerance:
                terminal_bonus = self.config.terminal_bonus
        
        # 总奖励
        total_reward = approach_reward + movement_reward + completion_bonus + time_penalty + terminal_bonus
        
        # 奖励详情
        breakdown = {
            'approach_reward': float(approach_reward),
            'movement_reward': float(movement_reward), 
            'completion_bonus': float(completion_bonus),
            'time_penalty': float(time_penalty),
            'terminal_bonus': float(terminal_bonus),
            'total_reward': float(total_reward),
            'target': target_info,
            'current_focus_user': self.current_focus_user,
            'visited_users': list(self.visited_users),
            'all_users_visited': self.all_users_visited,
            'user_close_time': dict(self.user_close_time)
        }
        
        return float(total_reward), breakdown
    
    def _calculate_user_approach_reward(self, uav_position, user_positions) -> float:
        """计算用户接近奖励（势函数增量）"""
        if self.current_focus_user is None or user_positions is None:
            return 0.0
        
        current_distance = self._get_user_distance(uav_position, self.current_focus_user, user_positions)
        
        if self.prev_focus_distance is not None:
            F_prev = self.potential_function(self.prev_focus_distance) 
            F_curr = self.potential_function(current_distance)
            approach_reward = self.config.w_approach * (F_curr - F_prev)
        else:
            approach_reward = 0.0
        
        self.prev_focus_distance = current_distance
        return approach_reward
    
    def _calculate_goal_approach_reward(self, uav_position, end_position) -> float:
        """计算目标接近奖励（势函数增量）"""
        current_distance = np.linalg.norm(uav_position - end_position)
        
        if self.prev_goal_distance is not None:
            F_prev = self.potential_function(self.prev_goal_distance)
            F_curr = self.potential_function(current_distance)
            approach_reward = self.config.w_goal * (F_curr - F_prev)
        else:
            approach_reward = 0.0
        
        self.prev_goal_distance = current_distance
        return approach_reward
    
    def _calculate_movement_reward(self, uav_position, prev_position) -> float:
        """计算移动奖励（鼓励探索，惩罚hover）"""
        if prev_position is None:
            return 0.0
        
        displacement = np.linalg.norm(uav_position - prev_position)
        
        if displacement < 1e-3:  # 基本没移动
            return -self.config.hover_penalty
        else:
            return self.config.movement_bonus * min(displacement / 10.0, 1.0)  # 奖励适度移动


# 测试简化奖励机制
def test_simple_reward():
    config = SimpleRewardConfig()
    calculator = SimpleRewardCalculator(config)
    
    # 测试场景
    uav_pos = np.array([0.0, 0.0, 50.0])
    end_pos = np.array([80.0, 80.0, 50.0])
    user_positions = np.array([[15.0, 75.0, 0.0], [75.0, 15.0, 0.0]])
    
    print("=== 简化奖励机制测试 ===")
    
    # 测试1：hover惩罚
    print("\n--- 测试1: Hover惩罚 ---")
    reward1, breakdown1 = calculator.calculate_reward(uav_pos, end_pos, user_positions)
    reward2, breakdown2 = calculator.calculate_reward(uav_pos, end_pos, user_positions, uav_pos)
    
    print(f"第1步奖励: {reward1:.4f}")
    print(f"第2步奖励(hover): {reward2:.4f}")
    print(f"移动奖励: {breakdown2['movement_reward']:.4f}")
    
    # 测试2：移动奖励
    print("\n--- 测试2: 移动奖励 ---")
    new_pos = np.array([5.0, 5.0, 50.0])  # 移动5m
    reward3, breakdown3 = calculator.calculate_reward(new_pos, end_pos, user_positions, uav_pos)
    
    print(f"移动后奖励: {reward3:.4f}")
    print(f"接近奖励: {breakdown3['approach_reward']:.4f}")
    print(f"移动奖励: {breakdown3['movement_reward']:.4f}")
    
    return calculator

if __name__ == "__main__":
    test_simple_reward()
