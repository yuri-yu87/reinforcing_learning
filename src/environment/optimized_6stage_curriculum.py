"""
优化的6阶段课程学习系统
确保阶段间平滑过渡，完成所有6个阶段
结合高级终点引导机制
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from .advanced_endpoint_guidance import AdvancedEndpointGuidanceConfig, AdvancedEndpointGuidanceCalculator


@dataclass
class Optimized6StageConfig:
    """
    优化的6阶段课程学习配置
    基于高级终点引导系统，确保每个阶段都能成功完成
    """
    
    # === 基础奖励权重 ===
    w_throughput_base: float = 100.0
    w_movement_bonus: float = 20.0
    
    # === 用户访问奖励（阶段自适应）===
    B_user_visit_base: float = 4000.0      # 基础单用户访问奖励
    B_user_visit_stage_multipliers: List[float] = None  # 各阶段用户访问奖励倍数
    B_all_users_visited: float = 10000.0   # 全用户访问完成奖励
    B_sequential_bonus: float = 3000.0     # 顺序访问奖励
    
    # === 终点奖励（阶段递增）===
    B_reach_end_base: float = 6000.0           # 基础到达终点奖励
    B_reach_end_stage_multipliers: List[float] = None  # 各阶段终点奖励倍数
    B_mission_complete_base: float = 15000.0    # 基础任务完成奖励
    
    # === 引导奖励（动态调整）===
    w_user_approach_base: float = 80.0
    w_end_approach_base: float = 200.0
    w_progress_bonus: float = 50.0
    
    # === 服务参数（阶段自适应）===
    user_service_radius_base: float = 60.0
    user_service_radius_stage_adjustments: List[float] = None  # 各阶段服务半径调整
    close_to_user_threshold_base: float = 80.0
    end_position_tolerance: float = 25.0
    user_visit_time_threshold: float = 1.0
    
    # === 时间约束 ===
    min_flight_time: float = 200.0
    max_flight_time: float = 300.0
    time_step: float = 0.1
    
    # === 课程学习专用参数 ===
    success_rate_thresholds: List[float] = None     # 各阶段成功率阈值
    min_episodes_per_stage: List[int] = None        # 各阶段最少训练回合数
    max_episodes_per_stage: List[int] = None        # 各阶段最大训练回合数
    
    # === 其他参数 ===
    stagnation_threshold: float = 1.0
    stagnation_time_window: float = 3.0
    w_stagnation: float = 3.0
    w_oob: float = 100.0
    
    def __post_init__(self):
        """初始化默认值"""
        if self.B_user_visit_stage_multipliers is None:
            # 递增的用户访问奖励倍数：后期阶段奖励更高
            self.B_user_visit_stage_multipliers = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0]
        
        if self.B_reach_end_stage_multipliers is None:
            # 递增的终点奖励倍数：后期阶段终点奖励更高
            self.B_reach_end_stage_multipliers = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        
        if self.user_service_radius_stage_adjustments is None:
            # 递减的服务半径：前期更宽松，后期更严格
            self.user_service_radius_stage_adjustments = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
        
        if self.success_rate_thresholds is None:
            # 递增的成功率要求：前期要求低，后期要求高
            self.success_rate_thresholds = [0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
        
        if self.min_episodes_per_stage is None:
            # 各阶段最少训练回合数：前期少，后期多
            self.min_episodes_per_stage = [15, 20, 25, 30, 35, 40]
        
        if self.max_episodes_per_stage is None:
            # 各阶段最大训练回合数：确保不会卡死
            self.max_episodes_per_stage = [30, 40, 50, 60, 70, 80]
    
    def get_stage_user_visit_reward(self, stage: int) -> float:
        """获取指定阶段的用户访问奖励"""
        stage_idx = max(0, min(stage - 1, len(self.B_user_visit_stage_multipliers) - 1))
        return self.B_user_visit_base * self.B_user_visit_stage_multipliers[stage_idx]
    
    def get_stage_end_reward(self, stage: int) -> float:
        """获取指定阶段的终点奖励"""
        stage_idx = max(0, min(stage - 1, len(self.B_reach_end_stage_multipliers) - 1))
        return self.B_reach_end_base * self.B_reach_end_stage_multipliers[stage_idx]
    
    def get_stage_service_radius(self, stage: int) -> float:
        """获取指定阶段的服务半径"""
        stage_idx = max(0, min(stage - 1, len(self.user_service_radius_stage_adjustments) - 1))
        return self.user_service_radius_base * self.user_service_radius_stage_adjustments[stage_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'B_user_visit_base': self.B_user_visit_base,
            'B_all_users_visited': self.B_all_users_visited,
            'B_reach_end_base': self.B_reach_end_base,
            'B_mission_complete_base': self.B_mission_complete_base,
            'user_service_radius_base': self.user_service_radius_base,
            'end_position_tolerance': self.end_position_tolerance,
            'success_rate_thresholds': self.success_rate_thresholds,
            'min_episodes_per_stage': self.min_episodes_per_stage,
            'max_episodes_per_stage': self.max_episodes_per_stage
        }


class Optimized6StageManager:
    """
    优化的6阶段课程学习管理器
    确保平滑过渡和成功完成所有阶段
    """
    
    def __init__(self, config: Optimized6StageConfig):
        self.config = config
        self.current_stage = 1
        self.stage_episodes = 0
        self.stage_successes = 0
        self.stage_rewards = []
        self.stage_completion_history = []
        
        # 性能追踪
        self.total_episodes = 0
        self.total_successes = 0
        self.best_stage_performance = {}
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取指定阶段的详细配置"""
        
        stage_configs = {
            1: {
                'stage_name': '阶段1：超近距离双用户学习',
                'description': '学会基本的起点→用户→终点模式',
                'user_positions': np.array([[15.0, 15.0, 0.0], [18.0, 18.0, 0.0]]),  # 两个极近距离用户
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'very_easy',
                'focus': 'basic_navigation',
                'expected_success_rate': 0.8
            },
            2: {
                'stage_name': '阶段2：近距离双用户学习',
                'description': '在适中距离学会稳定访问',
                'user_positions': np.array([[20.0, 20.0, 0.0], [25.0, 25.0, 0.0]]),  # 两个近距离用户
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'easy',
                'focus': 'stable_navigation',
                'expected_success_rate': 0.7
            },
            3: {
                'stage_name': '阶段3：超近距离双用户学习',
                'description': '学会访问两个相邻的用户',
                'user_positions': np.array([
                    [18.0, 20.0, 0.0],  # 用户1，很近
                    [22.0, 28.0, 0.0]   # 用户2，很近
                ]),
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'easy_medium',
                'focus': 'multi_user_basics',
                'expected_success_rate': 0.6
            },
            4: {
                'stage_name': '阶段4：近距离双用户学习',
                'description': '学会访问两个较近的用户',
                'user_positions': np.array([
                    [20.0, 30.0, 0.0],  # 用户1
                    [30.0, 40.0, 0.0]   # 用户2
                ]),
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'medium',
                'focus': 'multi_user_coordination',
                'expected_success_rate': 0.5
            },
            5: {
                'stage_name': '阶段5：中远距离双用户学习',
                'description': '学会处理中等复杂度的用户布局',
                'user_positions': np.array([
                    [25.0, 45.0, 0.0],  # 用户1
                    [45.0, 25.0, 0.0]   # 用户2，对角分布
                ]),
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'medium_hard',
                'focus': 'complex_routing',
                'expected_success_rate': 0.4
            },
            6: {
                'stage_name': '阶段6：完整场景挑战',
                'description': '在原始困难环境中完成完整任务',
                'user_positions': np.array([
                    [15.0, 75.0, 0.0],  # 用户1，原始困难位置
                    [75.0, 15.0, 0.0]   # 用户2，原始困难位置
                ]),
                'target_pattern': 'start → user1 → user2 → end',
                'difficulty_level': 'hard',
                'focus': 'full_capability',
                'expected_success_rate': 0.3
            }
        }
        
        base_config = stage_configs.get(stage, stage_configs[6])
        
        # 添加动态成功标准
        base_config['success_criteria'] = {
            'visit_all_users': True,
            'reach_end': True,
            'min_success_rate': self.config.success_rate_thresholds[stage - 1]
        }
        
        # 添加动态奖励倍数
        base_config['reward_multipliers'] = {
            'user_visit': self.config.B_user_visit_stage_multipliers[stage - 1],
            'end_reward': self.config.B_reach_end_stage_multipliers[stage - 1],
            'approach': 1.0,
            'completion': 1.0 + (stage - 1) * 0.2  # 后期阶段完成奖励更高
        }
        
        # 添加阶段特定参数
        base_config['stage_params'] = {
            'service_radius': self.config.get_stage_service_radius(stage),
            'min_episodes': self.config.min_episodes_per_stage[stage - 1],
            'max_episodes': self.config.max_episodes_per_stage[stage - 1]
        }
        
        return base_config
    
    def evaluate_stage_performance(self, episode_result: Dict[str, Any]) -> bool:
        """评估阶段性能并决定是否进入下一阶段"""
        self.stage_episodes += 1
        self.total_episodes += 1
        
        # 检查成功标准
        config = self.get_stage_config(self.current_stage)
        criteria = config['success_criteria']
        
        success = True
        if criteria.get('visit_all_users', False):
            success &= (episode_result.get('users_visited', 0) >= len(config['user_positions']))
        if criteria.get('reach_end', False):
            success &= episode_result.get('reached_end', False)
        
        if success:
            self.stage_successes += 1
            self.total_successes += 1
        
        self.stage_rewards.append(episode_result.get('total_reward', 0))
        
        # 动态阶段转换条件
        stage_params = config['stage_params']
        min_episodes = stage_params['min_episodes']
        max_episodes = stage_params['max_episodes']
        
        if self.stage_episodes >= min_episodes:
            success_rate = self.stage_successes / self.stage_episodes
            min_rate = criteria['min_success_rate']
            
            # 检查是否满足转换条件
            if success_rate >= min_rate:
                # 成功完成当前阶段
                self._record_stage_completion(True, success_rate)
                print(f"🎉 {config['stage_name']} 成功完成!")
                print(f"   成功率: {success_rate:.2%} (≥{min_rate:.2%})")
                print(f"   训练回合: {self.stage_episodes}")
                print(f"   平均奖励: {np.mean(self.stage_rewards[-10:]):.0f}")
                return True
            elif self.stage_episodes >= max_episodes:
                # 达到最大训练回合，强制转换
                self._record_stage_completion(False, success_rate)
                print(f"⚠️ {config['stage_name']} 达到最大训练回合，强制进入下一阶段")
                print(f"   当前成功率: {success_rate:.2%} (目标:{min_rate:.2%})")
                print(f"   建议: 下一阶段可能需要更多训练时间")
                return True
        
        # 提供实时反馈
        if self.stage_episodes % 5 == 0:
            current_success_rate = self.stage_successes / self.stage_episodes
            remaining_episodes = max_episodes - self.stage_episodes
            print(f"📊 {config['stage_name']} - 第{self.stage_episodes}回合")
            print(f"   当前成功率: {current_success_rate:.2%} (目标:{criteria['min_success_rate']:.2%})")
            print(f"   剩余最大回合: {remaining_episodes}")
        
        return False
    
    def advance_to_next_stage(self):
        """进入下一阶段"""
        if self.current_stage < 6:
            self.current_stage += 1
            self.stage_episodes = 0
            self.stage_successes = 0
            self.stage_rewards = []
            
            config = self.get_stage_config(self.current_stage)
            print(f"\n🚀 进入 {config['stage_name']}")
            print(f"   目标: {config['description']}")
            print(f"   难度: {config['difficulty_level']}")
            print(f"   模式: {config['target_pattern']}")
            print(f"   期望成功率: {config['expected_success_rate']:.1%}")
            
            # 显示阶段特定参数
            stage_params = config['stage_params']
            print(f"   服务半径: {stage_params['service_radius']:.1f}m")
            print(f"   训练范围: {stage_params['min_episodes']}-{stage_params['max_episodes']}回合")
        else:
            print("🏆 优化的6阶段课程学习完成！")
            self._print_completion_summary()
    
    def _record_stage_completion(self, success: bool, final_success_rate: float):
        """记录阶段完成情况"""
        completion_record = {
            'stage': self.current_stage,
            'success': success,
            'episodes': self.stage_episodes,
            'success_rate': final_success_rate,
            'avg_reward': np.mean(self.stage_rewards[-10:]) if self.stage_rewards else 0
        }
        self.stage_completion_history.append(completion_record)
        
        # 更新最佳性能记录
        stage_name = f"stage_{self.current_stage}"
        if stage_name not in self.best_stage_performance or final_success_rate > self.best_stage_performance[stage_name]['success_rate']:
            self.best_stage_performance[stage_name] = completion_record
    
    def _print_completion_summary(self):
        """打印完成总结"""
        print(f"\n📈 === 6阶段课程学习总结 === 📈")
        print(f"总训练回合: {self.total_episodes}")
        print(f"总成功次数: {self.total_successes}")
        print(f"总体成功率: {self.total_successes/max(1, self.total_episodes):.2%}")
        
        print(f"\n🏆 各阶段完成情况:")
        for record in self.stage_completion_history:
            status = "✅ 成功" if record['success'] else "⚠️ 强制"
            print(f"   阶段{record['stage']}: {status} | "
                  f"成功率 {record['success_rate']:.2%} | "
                  f"{record['episodes']}回合")
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """获取当前阶段信息"""
        config = self.get_stage_config(self.current_stage)
        success_rate = self.stage_successes / max(1, self.stage_episodes)
        
        return {
            'stage': self.current_stage,
            'stage_name': config['stage_name'],
            'episodes': self.stage_episodes,
            'successes': self.stage_successes,
            'success_rate': success_rate,
            'avg_reward': np.mean(self.stage_rewards[-10:]) if self.stage_rewards else 0,
            'user_positions': config['user_positions'],
            'target_pattern': config['target_pattern'],
            'difficulty_level': config['difficulty_level'],
            'expected_success_rate': config['expected_success_rate'],
            'stage_params': config['stage_params']
        }
    
    def is_curriculum_complete(self) -> bool:
        """检查课程学习是否完成"""
        return self.current_stage > 6


class Optimized6StageRewardCalculator:
    """
    优化的6阶段奖励计算器
    结合高级终点引导机制和阶段自适应参数
    """
    
    def __init__(self, config: Optimized6StageConfig, stage_manager: Optimized6StageManager):
        self.config = config
        self.stage_manager = stage_manager
        
        # 集成高级终点引导
        endpoint_config = AdvancedEndpointGuidanceConfig()
        self.endpoint_calculator = AdvancedEndpointGuidanceCalculator(endpoint_config, stage_manager)
        
        self.reset()
    
    def reset(self):
        """重置计算器状态"""
        self.endpoint_calculator.reset()
        
        # 阶段特定状态
        self.stage_performance_bonus = 0.0
        self.consecutive_successes = 0
    
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
        优化的6阶段奖励计算
        """
        # 获取当前阶段配置
        stage_config = self.stage_manager.get_stage_config(self.stage_manager.current_stage)
        stage_params = stage_config['stage_params']
        reward_multipliers = stage_config['reward_multipliers']
        
        # 使用高级终点引导计算基础奖励
        reward_breakdown = self.endpoint_calculator.calculate_reward(
            uav_position, end_position, stage_config['user_positions'], user_throughputs,
            current_time, current_speed, env_bounds, episode_done, reached_end
        )
        
        # === 阶段特定奖励调整 ===
        
        # 1. 用户访问奖励阶段调整
        if 'user_visit_bonus' in reward_breakdown and reward_breakdown['user_visit_bonus'] > 0:
            stage_multiplier = reward_multipliers['user_visit']
            reward_breakdown['user_visit_bonus'] *= stage_multiplier
            
            # 添加阶段进步奖励
            stage_progress_bonus = self.config.B_user_visit_base * 0.2 * self.stage_manager.current_stage
            reward_breakdown['stage_progress_bonus'] = stage_progress_bonus
        
        # 2. 终点奖励阶段调整
        if 'terminal_reach_end' in reward_breakdown and reward_breakdown['terminal_reach_end'] > 0:
            stage_multiplier = reward_multipliers['end_reward']
            reward_breakdown['terminal_reach_end'] *= stage_multiplier
        
        # 3. 阶段完成奖励
        if episode_done and reached_end:
            users_visited = len(self.endpoint_calculator.user_visited_flags)
            total_users = len(stage_config['user_positions'])
            
            if users_visited == total_users:
                # 阶段完成奖励
                stage_completion_bonus = self.config.B_mission_complete_base * reward_multipliers['completion']
                reward_breakdown['stage_completion_bonus'] = stage_completion_bonus
                
                # 困难度奖励：后期阶段完成给予更多奖励
                difficulty_bonus = self.config.B_mission_complete_base * 0.3 * self.stage_manager.current_stage
                reward_breakdown['difficulty_bonus'] = difficulty_bonus
        
        # 4. 服务半径自适应
        # 更新endpoint_calculator的配置以反映当前阶段的服务半径
        self.endpoint_calculator.config.user_service_radius = stage_params['service_radius']
        
        # === 重新计算总奖励 ===
        total_reward = sum(reward_breakdown.values())
        reward_breakdown['total'] = total_reward
        
        return reward_breakdown
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        base_stats = self.endpoint_calculator.get_stats()
        
        # 添加阶段特定统计
        base_stats.update({
            'current_stage': self.stage_manager.current_stage,
            'stage_episodes': self.stage_manager.stage_episodes,
            'stage_successes': self.stage_manager.stage_successes,
            'consecutive_successes': self.consecutive_successes,
            'stage_performance_bonus': self.stage_performance_bonus
        })
        
        return base_stats
