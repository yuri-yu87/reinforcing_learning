"""
智能奖励系统
实现动态权重调整、智能引导、完成度检测等高级奖励策略
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from .advanced_endpoint_guidance import AdvancedEndpointGuidanceConfig, AdvancedEndpointGuidanceCalculator
from .optimized_6stage_curriculum import Optimized6StageConfig, Optimized6StageManager


@dataclass
class IntelligentRewardConfig:
    """
    智能奖励系统配置
    """
    
    # === 动态权重调整参数 ===
    adaptive_weight_enabled: bool = True
    weight_adjustment_rate: float = 0.02  # 权重调整速率
    performance_window_size: int = 10     # 性能评估窗口大小
    
    # === 智能引导参数 ===
    intelligent_guidance_enabled: bool = True
    guidance_strength_multiplier: float = 2.0    # 引导强度倍数
    context_awareness_enabled: bool = True       # 上下文感知
    
    # === 完成度检测参数 ===
    completion_tracking_enabled: bool = True
    completion_bonus_multiplier: float = 1.5     # 完成度奖励倍数
    partial_completion_reward: bool = True       # 部分完成奖励
    
    # === 自适应学习参数 ===
    learning_difficulty_adjustment: bool = True
    difficulty_adaptation_rate: float = 0.01     # 难度自适应速率
    success_rate_target: float = 0.6             # 目标成功率
    
    # === 智能惩罚参数 ===
    smart_penalty_enabled: bool = True
    penalty_reduction_factor: float = 0.8        # 惩罚递减因子
    context_based_penalty: bool = True           # 基于上下文的惩罚
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'adaptive_weight_enabled': self.adaptive_weight_enabled,
            'intelligent_guidance_enabled': self.intelligent_guidance_enabled,
            'completion_tracking_enabled': self.completion_tracking_enabled,
            'learning_difficulty_adjustment': self.learning_difficulty_adjustment,
            'smart_penalty_enabled': self.smart_penalty_enabled
        }


class PerformanceTracker:
    """
    性能追踪器
    追踪和分析agent的学习表现
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """重置追踪器"""
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_completion_times = []
        self.episode_user_visit_rates = []
        self.episode_end_reach_rates = []
        
        # 实时性能指标
        self.current_success_rate = 0.0
        self.current_avg_reward = 0.0
        self.current_completion_rate = 0.0
        self.performance_trend = 0.0  # 正值表示改善，负值表示恶化
    
    def update(self, episode_result: Dict[str, Any]):
        """更新性能数据"""
        reward = episode_result.get('total_reward', 0)
        success = episode_result.get('reached_end', False) and episode_result.get('users_visited', 0) >= 1
        completion_time = episode_result.get('current_time', 0)
        users_visited = episode_result.get('users_visited', 0)
        total_users = episode_result.get('total_users', 2)
        
        # 添加到历史记录
        self.episode_rewards.append(reward)
        self.episode_successes.append(success)
        self.episode_completion_times.append(completion_time)
        self.episode_user_visit_rates.append(users_visited / max(1, total_users))
        self.episode_end_reach_rates.append(float(episode_result.get('reached_end', False)))
        
        # 保持窗口大小
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards = self.episode_rewards[-self.window_size:]
            self.episode_successes = self.episode_successes[-self.window_size:]
            self.episode_completion_times = self.episode_completion_times[-self.window_size:]
            self.episode_user_visit_rates = self.episode_user_visit_rates[-self.window_size:]
            self.episode_end_reach_rates = self.episode_end_reach_rates[-self.window_size:]
        
        # 更新实时指标
        self._update_current_metrics()
    
    def _update_current_metrics(self):
        """更新当前性能指标"""
        if len(self.episode_rewards) == 0:
            return
        
        # 当前成功率
        self.current_success_rate = np.mean(self.episode_successes)
        
        # 当前平均奖励
        self.current_avg_reward = np.mean(self.episode_rewards)
        
        # 当前完成率（用户访问+终点到达）
        visit_rate = np.mean(self.episode_user_visit_rates)
        end_rate = np.mean(self.episode_end_reach_rates)
        self.current_completion_rate = (visit_rate + end_rate) / 2.0
        
        # 性能趋势（最近5个 vs 之前5个）
        if len(self.episode_rewards) >= 10:
            recent_rewards = np.mean(self.episode_rewards[-5:])
            earlier_rewards = np.mean(self.episode_rewards[-10:-5])
            self.performance_trend = (recent_rewards - earlier_rewards) / max(abs(earlier_rewards), 1.0)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        return {
            'success_rate': self.current_success_rate,
            'avg_reward': self.current_avg_reward,
            'completion_rate': self.current_completion_rate,
            'performance_trend': self.performance_trend,
            'episodes_tracked': len(self.episode_rewards)
        }


class DynamicWeightAdjuster:
    """
    动态权重调整器
    根据学习表现自动调整奖励权重
    """
    
    def __init__(self, config: IntelligentRewardConfig):
        self.config = config
        self.base_weights = {}
        self.current_weights = {}
        self.adjustment_history = []
        
    def initialize_weights(self, base_config: Dict[str, float]):
        """初始化基础权重"""
        self.base_weights = base_config.copy()
        self.current_weights = base_config.copy()
    
    def adjust_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        """根据性能调整权重"""
        if not self.config.adaptive_weight_enabled:
            return self.current_weights.copy()
        
        adjustments = {}
        
        # 根据成功率调整
        success_rate = performance.get('success_rate', 0.0)
        target_rate = self.config.success_rate_target
        
        if success_rate < target_rate * 0.8:  # 成功率过低
            # 增强引导奖励，减少惩罚
            adjustments['user_approach_multiplier'] = 1.2
            adjustments['end_approach_multiplier'] = 1.3
            adjustments['penalty_reduction'] = 0.8
        elif success_rate > target_rate * 1.2:  # 成功率过高
            # 减少引导奖励，增加挑战
            adjustments['user_approach_multiplier'] = 0.9
            adjustments['end_approach_multiplier'] = 0.9
            adjustments['penalty_reduction'] = 1.0
        else:
            # 维持当前权重
            adjustments['user_approach_multiplier'] = 1.0
            adjustments['end_approach_multiplier'] = 1.0
            adjustments['penalty_reduction'] = 1.0
        
        # 根据完成度调整
        completion_rate = performance.get('completion_rate', 0.0)
        if completion_rate < 0.5:
            adjustments['completion_bonus_multiplier'] = 1.5
        else:
            adjustments['completion_bonus_multiplier'] = 1.0
        
        # 记录调整历史
        self.adjustment_history.append({
            'performance': performance.copy(),
            'adjustments': adjustments.copy()
        })
        
        return adjustments
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()


class ContextAwareGuidance:
    """
    上下文感知引导系统
    根据当前状态和历史行为提供智能引导
    """
    
    def __init__(self, config: IntelligentRewardConfig):
        self.config = config
        self.behavior_history = []
        self.stuck_patterns = []
        self.guidance_interventions = []
    
    def analyze_behavior(self, uav_position: np.ndarray, action_history: List[int],
                        user_positions: np.ndarray, end_position: np.ndarray) -> Dict[str, Any]:
        """分析UAV行为模式"""
        analysis = {
            'is_stuck': False,
            'stuck_type': None,
            'suggested_guidance': None,
            'intervention_strength': 1.0
        }
        
        if not self.config.context_awareness_enabled:
            return analysis
        
        # 检测停滞模式
        if len(self.behavior_history) >= 20:
            recent_positions = [record['position'] for record in self.behavior_history[-20:]]
            position_variance = np.var(recent_positions, axis=0)
            
            if np.sum(position_variance[:2]) < 5.0:  # 位置变化很小
                analysis['is_stuck'] = True
                analysis['stuck_type'] = 'position_stagnation'
                analysis['intervention_strength'] = 2.0
        
        # 检测循环行为
        if len(action_history) >= 10:
            recent_actions = action_history[-10:]
            if len(set(recent_actions)) <= 2:  # 只使用2种动作
                analysis['is_stuck'] = True
                analysis['stuck_type'] = 'action_repetition'
                analysis['intervention_strength'] = 1.5
        
        # 生成引导建议
        if analysis['is_stuck']:
            analysis['suggested_guidance'] = self._generate_guidance_suggestion(
                uav_position, user_positions, end_position, analysis['stuck_type']
            )
        
        # 记录行为
        self.behavior_history.append({
            'position': uav_position.copy(),
            'timestamp': len(self.behavior_history)
        })
        
        # 保持历史记录大小
        if len(self.behavior_history) > 50:
            self.behavior_history = self.behavior_history[-50:]
        
        return analysis
    
    def _generate_guidance_suggestion(self, uav_position: np.ndarray, 
                                    user_positions: np.ndarray, end_position: np.ndarray,
                                    stuck_type: str) -> Dict[str, Any]:
        """生成引导建议"""
        suggestion = {
            'type': 'exploration_bonus',
            'target': None,
            'strength': 1.0,
            'duration': 10
        }
        
        if stuck_type == 'position_stagnation':
            # 鼓励探索最近的未访问目标
            nearest_target = self._find_nearest_unvisited_target(uav_position, user_positions, end_position)
            suggestion['target'] = nearest_target
            suggestion['strength'] = 2.0
            suggestion['type'] = 'target_attraction'
        
        elif stuck_type == 'action_repetition':
            # 鼓励尝试不同的动作
            suggestion['type'] = 'action_diversity_bonus'
            suggestion['strength'] = 1.5
        
        return suggestion
    
    def _find_nearest_unvisited_target(self, uav_position: np.ndarray,
                                     user_positions: np.ndarray, end_position: np.ndarray) -> np.ndarray:
        """找到最近的未访问目标"""
        # 简化实现：返回最近的用户或终点
        all_targets = np.vstack([user_positions, end_position.reshape(1, -1)])
        distances = [np.linalg.norm(uav_position - target) for target in all_targets]
        nearest_idx = np.argmin(distances)
        return all_targets[nearest_idx]


class IntelligentRewardSystem:
    """
    智能奖励系统
    集成所有高级奖励策略
    """
    
    def __init__(self, config: IntelligentRewardConfig, 
                 stage_config: Optimized6StageConfig,
                 stage_manager: Optimized6StageManager):
        self.config = config
        self.stage_config = stage_config
        self.stage_manager = stage_manager
        
        # 初始化子系统
        self.performance_tracker = PerformanceTracker(config.performance_window_size)
        self.weight_adjuster = DynamicWeightAdjuster(config)
        self.context_guidance = ContextAwareGuidance(config)
        
        # 集成基础奖励计算器
        endpoint_config = AdvancedEndpointGuidanceConfig()
        self.base_calculator = AdvancedEndpointGuidanceCalculator(endpoint_config, stage_manager)
        
        # 初始化权重
        base_weights = {
            'throughput_weight': stage_config.w_throughput_base,
            'movement_weight': stage_config.w_movement_bonus,
            'user_approach_weight': stage_config.w_user_approach_base,
            'end_approach_weight': stage_config.w_end_approach_base,
            'progress_weight': stage_config.w_progress_bonus
        }
        self.weight_adjuster.initialize_weights(base_weights)
        
        # 状态追踪
        self.episode_count = 0
        self.action_history = []
        self.recent_interventions = []
    
    def reset(self):
        """重置奖励系统"""
        self.base_calculator.reset()
        self.action_history = []
        self.recent_interventions = []
    
    def calculate_intelligent_reward(self, 
                                   uav_position: np.ndarray,
                                   end_position: np.ndarray,
                                   user_positions: np.ndarray,
                                   user_throughputs: np.ndarray,
                                   current_time: float,
                                   current_speed: float,
                                   env_bounds: tuple,
                                   episode_done: bool,
                                   reached_end: bool,
                                   action: Optional[int] = None) -> Dict[str, float]:
        """
        智能奖励计算
        """
        # 获取当前阶段配置
        stage_config = self.stage_manager.get_stage_config(self.stage_manager.current_stage)
        effective_user_positions = stage_config['user_positions']
        
        # 基础奖励计算
        reward_breakdown = self.base_calculator.calculate_reward(
            uav_position, end_position, effective_user_positions, user_throughputs,
            current_time, current_speed, env_bounds, episode_done, reached_end
        )
        
        # === 动态权重调整 ===
        if self.config.adaptive_weight_enabled:
            performance = self.performance_tracker.get_performance_summary()
            weight_adjustments = self.weight_adjuster.adjust_weights(performance)
            
            # 应用权重调整
            if 'user_approach' in reward_breakdown:
                reward_breakdown['user_approach'] *= weight_adjustments.get('user_approach_multiplier', 1.0)
            if 'end_approach' in reward_breakdown:
                reward_breakdown['end_approach'] *= weight_adjustments.get('end_approach_multiplier', 1.0)
        
        # === 上下文感知引导 ===
        if self.config.intelligent_guidance_enabled and action is not None:
            self.action_history.append(action)
            if len(self.action_history) > 20:
                self.action_history = self.action_history[-20:]
            
            # 行为分析
            behavior_analysis = self.context_guidance.analyze_behavior(
                uav_position, self.action_history, effective_user_positions, end_position
            )
            
            # 应用智能引导
            if behavior_analysis['is_stuck']:
                guidance_bonus = self._apply_intelligent_guidance(
                    reward_breakdown, behavior_analysis, uav_position, effective_user_positions, end_position
                )
                reward_breakdown['intelligent_guidance_bonus'] = guidance_bonus
        
        # === 完成度检测和奖励 ===
        if self.config.completion_tracking_enabled:
            completion_bonus = self._calculate_completion_bonus(
                uav_position, end_position, effective_user_positions, episode_done, reached_end
            )
            reward_breakdown['completion_bonus'] = completion_bonus
        
        # === 智能惩罚调整 ===
        if self.config.smart_penalty_enabled:
            self._adjust_intelligent_penalties(reward_breakdown)
        
        # === 学习难度自适应 ===
        if self.config.learning_difficulty_adjustment:
            difficulty_adjustment = self._calculate_difficulty_adjustment()
            reward_breakdown['difficulty_adjustment'] = difficulty_adjustment
        
        # 重新计算总奖励
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total'] = total_reward
        
        return reward_breakdown
    
    def _apply_intelligent_guidance(self, reward_breakdown: Dict[str, float],
                                  behavior_analysis: Dict[str, Any],
                                  uav_position: np.ndarray,
                                  user_positions: np.ndarray,
                                  end_position: np.ndarray) -> float:
        """应用智能引导"""
        guidance_bonus = 0.0
        suggestion = behavior_analysis.get('suggested_guidance', {})
        
        if suggestion.get('type') == 'target_attraction':
            # 目标吸引奖励
            target = suggestion.get('target')
            if target is not None:
                distance = np.linalg.norm(uav_position - target)
                attraction_bonus = suggestion.get('strength', 1.0) * 100.0 * (1.0 / (1.0 + distance / 50.0))
                guidance_bonus += attraction_bonus
        
        elif suggestion.get('type') == 'action_diversity_bonus':
            # 动作多样性奖励
            if len(set(self.action_history[-10:])) > 2:
                diversity_bonus = suggestion.get('strength', 1.0) * 50.0
                guidance_bonus += diversity_bonus
        
        elif suggestion.get('type') == 'exploration_bonus':
            # 探索奖励
            exploration_bonus = suggestion.get('strength', 1.0) * 30.0
            guidance_bonus += exploration_bonus
        
        # 记录干预
        if guidance_bonus > 0:
            self.recent_interventions.append({
                'type': suggestion.get('type'),
                'bonus': guidance_bonus,
                'timestamp': len(self.recent_interventions)
            })
            
            if len(self.recent_interventions) > 10:
                self.recent_interventions = self.recent_interventions[-10:]
        
        return guidance_bonus
    
    def _calculate_completion_bonus(self, uav_position: np.ndarray, end_position: np.ndarray,
                                  user_positions: np.ndarray, episode_done: bool, 
                                  reached_end: bool) -> float:
        """计算完成度奖励"""
        completion_bonus = 0.0
        
        # 用户访问完成度
        users_visited = len(self.base_calculator.user_visited_flags)
        total_users = len(user_positions)
        visit_completion = users_visited / max(1, total_users)
        
        # 终点接近完成度
        distance_to_end = np.linalg.norm(uav_position - end_position)
        end_completion = max(0, 1.0 - distance_to_end / 100.0)
        
        # 整体完成度
        overall_completion = (visit_completion + end_completion) / 2.0
        
        # 部分完成奖励
        if self.config.partial_completion_reward:
            partial_bonus = overall_completion * 200.0 * self.config.completion_bonus_multiplier
            completion_bonus += partial_bonus
        
        # 完整完成奖励
        if episode_done and reached_end and users_visited == total_users:
            full_completion_bonus = 1000.0 * self.config.completion_bonus_multiplier
            completion_bonus += full_completion_bonus
        
        return completion_bonus
    
    def _adjust_intelligent_penalties(self, reward_breakdown: Dict[str, float]):
        """智能惩罚调整"""
        # 获取当前性能
        performance = self.performance_tracker.get_performance_summary()
        
        # 如果学习困难，减少惩罚
        if performance.get('success_rate', 0.0) < 0.3:
            penalty_keys = [key for key in reward_breakdown.keys() if 'penalty' in key]
            for key in penalty_keys:
                if reward_breakdown[key] < 0:
                    reward_breakdown[key] *= self.config.penalty_reduction_factor
    
    def _calculate_difficulty_adjustment(self) -> float:
        """计算难度调整"""
        performance = self.performance_tracker.get_performance_summary()
        target_rate = self.config.success_rate_target
        current_rate = performance.get('success_rate', 0.0)
        
        # 如果成功率过低，给予额外支持
        if current_rate < target_rate * 0.5:
            return 200.0  # 额外支持奖励
        elif current_rate < target_rate * 0.8:
            return 100.0  # 中等支持奖励
        else:
            return 0.0    # 无额外支持
    
    def update_episode_performance(self, episode_result: Dict[str, Any]):
        """更新回合性能"""
        self.episode_count += 1
        
        # 添加系统特定信息
        episode_result['total_users'] = len(self.stage_manager.get_stage_config(
            self.stage_manager.current_stage
        )['user_positions'])
        
        self.performance_tracker.update(episode_result)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        base_stats = self.base_calculator.get_stats()
        performance_stats = self.performance_tracker.get_performance_summary()
        
        return {
            'base_stats': base_stats,
            'performance_stats': performance_stats,
            'episode_count': self.episode_count,
            'current_weights': self.weight_adjuster.get_current_weights(),
            'recent_interventions': len(self.recent_interventions),
            'system_config': self.config.to_dict()
        }
