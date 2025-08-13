"""
完整的6阶段高级训练系统
集成所有优化组件：高级终点引导、优化课程学习、智能奖励系统
确保能够完成所有6个阶段
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保src模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment.uav_env import UAVEnvironment
from environment.advanced_endpoint_guidance import AdvancedEndpointGuidanceConfig, AdvancedEndpointGuidanceCalculator
from environment.optimized_6stage_curriculum import Optimized6StageConfig, Optimized6StageManager, Optimized6StageRewardCalculator
from environment.intelligent_reward_system import IntelligentRewardConfig, IntelligentRewardSystem


class Advanced6StageCallback(BaseCallback):
    """
    高级6阶段训练回调
    """
    
    def __init__(self, stage_manager: Optimized6StageManager, 
                 intelligent_system: IntelligentRewardSystem, verbose: int = 1):
        super().__init__(verbose)
        self.stage_manager = stage_manager
        self.intelligent_system = intelligent_system
        
        # 训练统计
        self.episode_count = 0
        self.stage_history = []
        self.success_history = []
        self.reward_history = []
        
        # 性能追踪
        self.training_start_time = None
        self.stage_start_times = {}
        self.best_performance = {}
        
    def _on_training_start(self) -> None:
        """训练开始时调用"""
        self.training_start_time = time.time()
        self.stage_start_times[self.stage_manager.current_stage] = time.time()
        
        print("🚀 === 启动完整6阶段高级训练系统 === 🚀")
        print(f"集成组件: 高级终点引导 + 优化课程学习 + 智能奖励系统")
        print(f"目标: 完成所有6个阶段的课程学习")
        
        current_stage_info = self.stage_manager.get_current_stage_info()
        print(f"\n📍 开始阶段: {current_stage_info['stage_name']}")
        print(f"   目标: {current_stage_info.get('description', 'N/A')}")
        print(f"   用户位置: {current_stage_info['user_positions'][:, :2].tolist()}")
        print(f"   期望成功率: {current_stage_info.get('expected_success_rate', 0):.1%}")
    
    def _on_step(self) -> bool:
        """每步调用"""
        return True
    
    def _on_rollout_end(self) -> None:
        """每个rollout结束时调用"""
        # 获取最新回合信息
        if hasattr(self.training_env, 'get_episode_rewards'):
            episode_rewards = self.training_env.get_episode_rewards()
            if len(episode_rewards) > len(self.reward_history):
                # 新回合完成
                self.episode_count += 1
                latest_reward = episode_rewards[-1]
                self.reward_history.append(latest_reward)
                
                # 获取回合结果（从environment info中）
                if hasattr(self.training_env.unwrapped, '_get_info'):
                    info = self.training_env.unwrapped._get_info()
                    self._process_episode_result(info, latest_reward)
    
    def _process_episode_result(self, info: Dict[str, Any], reward: float):
        """处理回合结果"""
        # 构建回合结果
        episode_result = {
            'total_reward': reward,
            'users_visited': info.get('users_visited', 0),
            'reached_end': info.get('reached_end', False),
            'current_time': info.get('current_time', 0),
            'uav_position': info.get('uav_position', [0, 0, 0])
        }
        
        # 更新智能系统性能
        self.intelligent_system.update_episode_performance(episode_result)
        
        # 检查任务成功
        success = (episode_result['users_visited'] >= 1 and episode_result['reached_end'])
        self.success_history.append(success)
        
        # 记录阶段信息
        current_stage = self.stage_manager.current_stage
        self.stage_history.append(current_stage)
        
        # 评估是否需要进入下一阶段
        should_advance = self.stage_manager.evaluate_stage_performance(episode_result)
        
        if should_advance:
            self._handle_stage_advancement()
        
        # 实时反馈（每10回合）
        if self.episode_count % 10 == 0:
            self._print_progress_update()
    
    def _handle_stage_advancement(self):
        """处理阶段晋级"""
        completed_stage = self.stage_manager.current_stage
        stage_time = time.time() - self.stage_start_times.get(completed_stage, time.time())
        
        # 记录阶段性能
        stage_info = self.stage_manager.get_current_stage_info()
        stage_performance = {
            'stage': completed_stage,
            'episodes': stage_info['episodes'],
            'success_rate': stage_info['success_rate'],
            'training_time': stage_time,
            'avg_reward': stage_info['avg_reward']
        }
        self.best_performance[f'stage_{completed_stage}'] = stage_performance
        
        # 更新环境用户位置
        if not self.stage_manager.is_curriculum_complete():
            self.stage_manager.advance_to_next_stage()
            new_stage_config = self.stage_manager.get_stage_config(self.stage_manager.current_stage)
            
            # 更新环境
            if hasattr(self.training_env.unwrapped, 'user_manager'):
                self.training_env.unwrapped.user_manager.set_user_positions(
                    new_stage_config['user_positions']
                )
            
            # 记录新阶段开始时间
            self.stage_start_times[self.stage_manager.current_stage] = time.time()
        else:
            # 课程学习完成
            self._handle_curriculum_completion()
    
    def _handle_curriculum_completion(self):
        """处理课程学习完成"""
        total_time = time.time() - self.training_start_time
        
        print(f"\n🏆 === 6阶段课程学习完成！=== 🏆")
        print(f"总训练时间: {total_time/60:.1f}分钟")
        print(f"总训练回合: {self.episode_count}")
        print(f"总体成功率: {np.mean(self.success_history[-50:]):.2%}")
        
        # 生成完成报告
        self._generate_completion_report()
    
    def _print_progress_update(self):
        """打印进度更新"""
        current_stage_info = self.stage_manager.get_current_stage_info()
        recent_success_rate = np.mean(self.success_history[-10:]) if len(self.success_history) >= 10 else 0
        recent_reward = np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        
        print(f"\n📊 === 第{self.episode_count}回合进度报告 === 📊")
        print(f"当前阶段: {current_stage_info['stage_name']} ({current_stage_info['stage']}/6)")
        print(f"阶段进度: {current_stage_info['episodes']}回合 | 成功率: {current_stage_info['success_rate']:.2%}")
        print(f"最近10回合: 成功率 {recent_success_rate:.2%} | 平均奖励 {recent_reward:.0f}")
        
        # 智能系统状态
        system_stats = self.intelligent_system.get_system_stats()
        performance_stats = system_stats['performance_stats']
        print(f"智能系统: 成功率 {performance_stats['success_rate']:.2%} | "
              f"趋势 {performance_stats['performance_trend']:+.2f} | "
              f"干预次数 {system_stats['recent_interventions']}")
    
    def _generate_completion_report(self):
        """生成完成报告"""
        report_file = f"6stage_completion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 6阶段课程学习完成报告 ===\n\n")
            f.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练回合: {self.episode_count}\n")
            f.write(f"总体成功率: {np.mean(self.success_history):.2%}\n\n")
            
            f.write("各阶段性能:\n")
            for stage_name, performance in self.best_performance.items():
                f.write(f"  {stage_name}: {performance['episodes']}回合, "
                       f"成功率{performance['success_rate']:.2%}, "
                       f"训练时间{performance['training_time']/60:.1f}分钟\n")
            
            f.write(f"\n报告保存位置: {os.path.abspath(report_file)}")
        
        print(f"📄 完成报告已保存: {report_file}")


def create_advanced_6stage_environment() -> Tuple[UAVEnvironment, Optimized6StageManager, IntelligentRewardSystem]:
    """
    创建高级6阶段环境
    """
    print("🛠️ 创建高级6阶段环境...")
    
    # 1. 创建配置
    stage_config = Optimized6StageConfig()
    intelligent_config = IntelligentRewardConfig()
    
    # 2. 创建阶段管理器
    stage_manager = Optimized6StageManager(stage_config)
    
    # 3. 创建智能奖励系统
    intelligent_system = IntelligentRewardSystem(intelligent_config, stage_config, stage_manager)
    
    # 4. 创建环境
    env = UAVEnvironment(
        env_size=(100, 100, 50),
        num_users=2,
        num_antennas=8,
        start_position=(0, 0, 50),
        end_position=(80, 80, 50),
        flight_time=250.0,
        time_step=0.1,
        transmit_power=0.5,
        max_speed=30.0,
        min_speed=10.0,
        fixed_users=True,
        seed=42
    )
    
    # 5. 设置第一阶段的用户位置
    stage1_config = stage_manager.get_stage_config(1)
    env.user_manager.set_user_positions(stage1_config['user_positions'])
    
    # 6. 集成智能奖励系统到环境
    class IntelligentRewardCalculator:
        def __init__(self, intelligent_system):
            self.intelligent_system = intelligent_system
            
        def calculate_reward(self, **kwargs):
            return self.intelligent_system.calculate_intelligent_reward(**kwargs)
        
        def reset(self):
            self.intelligent_system.reset()
        
        def get_stats(self):
            return self.intelligent_system.get_system_stats()
    
    env.set_reward_calculator(IntelligentRewardCalculator(intelligent_system))
    
    print("✅ 高级6阶段环境创建完成")
    print(f"   - 阶段管理器: {type(stage_manager).__name__}")
    print(f"   - 智能奖励系统: {type(intelligent_system).__name__}")
    print(f"   - 集成组件: 终点引导 + 课程学习 + 智能奖励")
    
    return env, stage_manager, intelligent_system


def create_enhanced_dqn_agent(env: UAVEnvironment) -> DQN:
    """
    创建增强的DQN智能体
    """
    print("🤖 创建增强DQN智能体...")
    
    agent = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=5e-4,           # 优化学习率
        gamma=0.995,                  # 高折扣因子
        batch_size=128,               # 大批次训练
        buffer_size=400000,           # 大经验缓冲区
        exploration_initial_eps=0.9,  # 初始探索率
        exploration_final_eps=0.05,   # 最终探索率
        exploration_fraction=0.7,     # 探索衰减比例
        learning_starts=2000,         # 开始学习的步数
        train_freq=4,                 # 训练频率
        target_update_interval=1000,  # 目标网络更新间隔
        policy_kwargs=dict(
            net_arch=[512, 256, 128, 64]  # 深度网络架构
        ),
        verbose=1,
        seed=42
    )
    
    print("✅ 增强DQN智能体创建完成")
    print(f"   - 网络架构: [512, 256, 128, 64]")
    print(f"   - 学习率: {agent.learning_rate}")
    print(f"   - 缓冲区大小: {agent.buffer_size}")
    print(f"   - 探索策略: {agent.exploration_initial_eps} → {agent.exploration_final_eps}")
    
    return agent


def train_complete_6stage_system(total_timesteps: int = 200000):
    """
    训练完整的6阶段系统
    """
    print("🎯 === 开始完整6阶段高级训练 === 🎯")
    
    # 1. 创建环境和系统
    env, stage_manager, intelligent_system = create_advanced_6stage_environment()
    monitored_env = Monitor(env)
    
    # 2. 创建智能体
    agent = create_enhanced_dqn_agent(monitored_env)
    
    # 3. 创建高级回调
    callback = Advanced6StageCallback(stage_manager, intelligent_system, verbose=1)
    
    # 4. 开始训练
    print(f"\n🚀 开始训练，目标步数: {total_timesteps:,}")
    print(f"预期训练时间: {total_timesteps/10000:.0f}-{total_timesteps/5000:.0f}分钟")
    
    start_time = time.time()
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成，耗时: {training_time/60:.1f}分钟")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        training_time = time.time() - start_time
        print(f"已训练时间: {training_time/60:.1f}分钟")
    
    # 5. 保存模型
    model_path = f"advanced_6stage_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agent.save(model_path)
    print(f"💾 模型已保存: {model_path}")
    
    # 6. 生成训练报告
    generate_training_report(callback, stage_manager, intelligent_system, training_time)
    
    return agent, monitored_env, callback


def generate_training_report(callback: Advanced6StageCallback, 
                           stage_manager: Optimized6StageManager,
                           intelligent_system: IntelligentRewardSystem,
                           training_time: float):
    """
    生成训练报告
    """
    print(f"\n📊 === 生成训练报告 === 📊")
    
    # 创建结果目录
    results_dir = f"results/6stage_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 训练曲线图
    if len(callback.success_history) > 0:
        plot_training_curves(callback, results_dir)
    
    # 2. 阶段性能图
    plot_stage_performance(callback, stage_manager, results_dir)
    
    # 3. 智能系统分析
    plot_intelligent_system_analysis(intelligent_system, results_dir)
    
    # 4. 综合报告
    generate_comprehensive_report(callback, stage_manager, intelligent_system, 
                                training_time, results_dir)
    
    print(f"📁 训练报告已保存到: {results_dir}")


def plot_training_curves(callback: Advanced6StageCallback, results_dir: str):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('6阶段训练曲线分析', fontsize=16, fontweight='bold')
    
    # 成功率曲线
    if len(callback.success_history) > 0:
        window_size = min(20, len(callback.success_history) // 10)
        success_rate_smooth = []
        for i in range(window_size, len(callback.success_history)):
            success_rate_smooth.append(np.mean(callback.success_history[i-window_size:i]))
        
        axes[0, 0].plot(success_rate_smooth, 'b-', linewidth=2)
        axes[0, 0].set_title('成功率变化曲线')
        axes[0, 0].set_xlabel('训练回合')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
    
    # 奖励曲线
    if len(callback.reward_history) > 0:
        window_size = min(20, len(callback.reward_history) // 10)
        reward_smooth = []
        for i in range(window_size, len(callback.reward_history)):
            reward_smooth.append(np.mean(callback.reward_history[i-window_size:i]))
        
        axes[0, 1].plot(reward_smooth, 'g-', linewidth=2)
        axes[0, 1].set_title('奖励变化曲线')
        axes[0, 1].set_xlabel('训练回合')
        axes[0, 1].set_ylabel('平均奖励')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 阶段分布
    if len(callback.stage_history) > 0:
        stage_counts = {}
        for stage in callback.stage_history:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        stages = list(stage_counts.keys())
        counts = list(stage_counts.values())
        
        axes[1, 0].bar([f'阶段{s}' for s in stages], counts, color='orange', alpha=0.7)
        axes[1, 0].set_title('各阶段训练分布')
        axes[1, 0].set_ylabel('训练回合数')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 阶段成功率对比
    if hasattr(callback, 'best_performance') and callback.best_performance:
        stages = []
        success_rates = []
        
        for stage_name, performance in callback.best_performance.items():
            stages.append(stage_name.replace('stage_', '阶段'))
            success_rates.append(performance['success_rate'])
        
        if stages:
            axes[1, 1].bar(stages, success_rates, color='purple', alpha=0.7)
            axes[1, 1].set_title('各阶段最终成功率')
            axes[1, 1].set_ylabel('成功率')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_stage_performance(callback: Advanced6StageCallback, 
                         stage_manager: Optimized6StageManager, results_dir: str):
    """绘制阶段性能图"""
    if not hasattr(callback, 'best_performance') or not callback.best_performance:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('6阶段性能详细分析', fontsize=16, fontweight='bold')
    
    stages = []
    episodes = []
    success_rates = []
    avg_rewards = []
    training_times = []
    
    for stage_name, performance in callback.best_performance.items():
        stages.append(int(stage_name.split('_')[1]))
        episodes.append(performance['episodes'])
        success_rates.append(performance['success_rate'])
        avg_rewards.append(performance['avg_reward'])
        training_times.append(performance['training_time'] / 60)  # 转换为分钟
    
    if stages:
        # 各阶段训练回合数
        axes[0, 0].plot(stages, episodes, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('各阶段训练回合数')
        axes[0, 0].set_xlabel('阶段')
        axes[0, 0].set_ylabel('训练回合数')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 各阶段成功率
        axes[0, 1].plot(stages, success_rates, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_title('各阶段成功率')
        axes[0, 1].set_xlabel('阶段')
        axes[0, 1].set_ylabel('成功率')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 各阶段平均奖励
        axes[1, 0].plot(stages, avg_rewards, 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('各阶段平均奖励')
        axes[1, 0].set_xlabel('阶段')
        axes[1, 0].set_ylabel('平均奖励')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 各阶段训练时间
        axes[1, 1].plot(stages, training_times, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('各阶段训练时间')
        axes[1, 1].set_xlabel('阶段')
        axes[1, 1].set_ylabel('训练时间 (分钟)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'stage_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_intelligent_system_analysis(intelligent_system: IntelligentRewardSystem, results_dir: str):
    """绘制智能系统分析图"""
    system_stats = intelligent_system.get_system_stats()
    performance_stats = system_stats['performance_stats']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('智能奖励系统分析', fontsize=16, fontweight='bold')
    
    # 性能指标雷达图（简化版）
    metrics = ['成功率', '完成率', '性能趋势']
    values = [
        performance_stats.get('success_rate', 0),
        performance_stats.get('completion_rate', 0),
        max(0, performance_stats.get('performance_trend', 0))  # 只显示正趋势
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    ax1.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax1.fill(angles, values, alpha=0.25, color='blue')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_title('系统性能雷达图')
    ax1.set_ylim(0, 1)
    
    # 权重调整历史（如果有的话）
    current_weights = system_stats.get('current_weights', {})
    if current_weights:
        weight_names = list(current_weights.keys())
        weight_values = list(current_weights.values())
        
        ax2.bar(range(len(weight_names)), weight_values, alpha=0.7, color='green')
        ax2.set_title('当前权重分布')
        ax2.set_xticks(range(len(weight_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in weight_names], rotation=45)
        ax2.set_ylabel('权重值')
    
    # 系统配置显示
    config_info = system_stats.get('system_config', {})
    config_text = "智能系统配置:\n"
    for key, value in config_info.items():
        config_text += f"• {key}: {value}\n"
    
    ax3.text(0.05, 0.95, config_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax3.set_title('系统配置信息')
    ax3.axis('off')
    
    # 干预统计
    base_stats = system_stats.get('base_stats', {})
    intervention_info = f"""
    智能干预统计:
    • 总回合数: {system_stats.get('episode_count', 0)}
    • 近期干预: {system_stats.get('recent_interventions', 0)}
    • 用户访问: {len(base_stats.get('user_visited_flags', []))}
    • 引导激活: {base_stats.get('end_guidance_activated', False)}
    • 强引导: {base_stats.get('strong_guidance_activated', False)}
    """
    
    ax4.text(0.05, 0.95, intervention_info, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax4.set_title('智能干预统计')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'intelligent_system_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_report(callback: Advanced6StageCallback,
                                stage_manager: Optimized6StageManager,
                                intelligent_system: IntelligentRewardSystem,
                                training_time: float, results_dir: str):
    """生成综合报告"""
    report_path = os.path.join(results_dir, 'comprehensive_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 6阶段高级训练系统综合报告 ===\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练总时间: {training_time/60:.1f}分钟\n")
        f.write(f"总训练回合: {callback.episode_count}\n\n")
        
        # 整体性能
        if len(callback.success_history) > 0:
            f.write("=== 整体性能 ===\n")
            f.write(f"总体成功率: {np.mean(callback.success_history):.2%}\n")
            f.write(f"最终成功率: {np.mean(callback.success_history[-20:]):.2%}\n")
            f.write(f"平均奖励: {np.mean(callback.reward_history):.0f}\n")
            f.write(f"最终平均奖励: {np.mean(callback.reward_history[-20:]):.0f}\n\n")
        
        # 各阶段详情
        f.write("=== 各阶段详细性能 ===\n")
        if hasattr(callback, 'best_performance'):
            for stage_name, performance in callback.best_performance.items():
                stage_num = stage_name.split('_')[1]
                f.write(f"阶段{stage_num}:\n")
                f.write(f"  训练回合: {performance['episodes']}\n")
                f.write(f"  成功率: {performance['success_rate']:.2%}\n")
                f.write(f"  平均奖励: {performance['avg_reward']:.0f}\n")
                f.write(f"  训练时间: {performance['training_time']/60:.1f}分钟\n\n")
        
        # 智能系统分析
        system_stats = intelligent_system.get_system_stats()
        f.write("=== 智能奖励系统分析 ===\n")
        performance_stats = system_stats.get('performance_stats', {})
        f.write(f"系统成功率: {performance_stats.get('success_rate', 0):.2%}\n")
        f.write(f"完成率: {performance_stats.get('completion_rate', 0):.2%}\n")
        f.write(f"性能趋势: {performance_stats.get('performance_trend', 0):+.3f}\n")
        f.write(f"监控回合: {performance_stats.get('episodes_tracked', 0)}\n")
        f.write(f"智能干预: {system_stats.get('recent_interventions', 0)}次\n\n")
        
        # 系统配置
        f.write("=== 系统配置 ===\n")
        config_info = system_stats.get('system_config', {})
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\n=== 文件位置 ===\n")
        f.write(f"报告目录: {os.path.abspath(results_dir)}\n")
        f.write(f"训练曲线: training_curves.png\n")
        f.write(f"阶段性能: stage_performance.png\n")
        f.write(f"智能分析: intelligent_system_analysis.png\n")
    
    print(f"📄 综合报告已保存: {report_path}")


def evaluate_final_performance(agent: DQN, env: UAVEnvironment, 
                             stage_manager: Optimized6StageManager,
                             num_episodes: int = 10) -> Dict[str, Any]:
    """
    评估最终性能
    """
    print(f"\n🔍 === 最终性能评估 ({num_episodes}回合) === 🔍")
    
    evaluation_results = {
        'stage_results': {},
        'overall_performance': {}
    }
    
    # 在每个阶段评估性能
    for stage in range(1, 7):
        stage_config = stage_manager.get_stage_config(stage)
        env.user_manager.set_user_positions(stage_config['user_positions'])
        
        stage_successes = 0
        stage_rewards = []
        
        print(f"\n📍 评估{stage_config['stage_name']}...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 3000:
                action, _ = agent.predict(obs, deterministic=True)
                action = int(np.asarray(action).ravel()[0])  # 确保action是整数
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1
            
            # 检查成功
            success = (info.get('users_visited', 0) >= len(stage_config['user_positions']) and 
                      info.get('reached_end', False))
            
            if success:
                stage_successes += 1
            stage_rewards.append(total_reward)
        
        # 记录阶段结果
        stage_success_rate = stage_successes / num_episodes
        stage_avg_reward = np.mean(stage_rewards)
        
        evaluation_results['stage_results'][stage] = {
            'success_rate': stage_success_rate,
            'avg_reward': stage_avg_reward,
            'successes': stage_successes,
            'total_episodes': num_episodes
        }
        
        print(f"   成功率: {stage_success_rate:.2%}")
        print(f"   平均奖励: {stage_avg_reward:.0f}")
    
    # 计算整体性能
    all_success_rates = [result['success_rate'] for result in evaluation_results['stage_results'].values()]
    all_avg_rewards = [result['avg_reward'] for result in evaluation_results['stage_results'].values()]
    
    evaluation_results['overall_performance'] = {
        'avg_success_rate': np.mean(all_success_rates),
        'min_success_rate': np.min(all_success_rates),
        'max_success_rate': np.max(all_success_rates),
        'avg_reward': np.mean(all_avg_rewards),
        'stages_above_30_percent': sum(1 for rate in all_success_rates if rate >= 0.3),
        'stages_above_50_percent': sum(1 for rate in all_success_rates if rate >= 0.5)
    }
    
    print(f"\n🏆 === 整体评估结果 === 🏆")
    print(f"平均成功率: {evaluation_results['overall_performance']['avg_success_rate']:.2%}")
    print(f"成功率范围: {evaluation_results['overall_performance']['min_success_rate']:.2%} - {evaluation_results['overall_performance']['max_success_rate']:.2%}")
    print(f"≥30%成功率阶段: {evaluation_results['overall_performance']['stages_above_30_percent']}/6")
    print(f"≥50%成功率阶段: {evaluation_results['overall_performance']['stages_above_50_percent']}/6")
    
    return evaluation_results


def main():
    """
    主函数
    """
    print("🌟 === 完整6阶段高级训练系统 === 🌟")
    print("集成功能:")
    print("  ✅ 高级终点引导机制")
    print("  ✅ 优化6阶段课程学习")
    print("  ✅ 智能奖励系统")
    print("  ✅ 动态权重调整")
    print("  ✅ 上下文感知引导")
    print("  ✅ 完成度检测")
    print("  ✅ 智能惩罚调整")
    
    # 询问训练配置
    print(f"\n⚙️ 训练配置选择:")
    print("1. 快速测试 (50,000步, ~10分钟)")
    print("2. 标准训练 (200,000步, ~40分钟)")
    print("3. 完整训练 (500,000步, ~100分钟)")
    print("4. 自定义")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == '1':
        timesteps = 50000
    elif choice == '2':
        timesteps = 200000
    elif choice == '3':
        timesteps = 500000
    elif choice == '4':
        timesteps = int(input("请输入训练步数: "))
    else:
        timesteps = 200000
        print("使用默认配置: 200,000步")
    
    # 开始训练
    agent, env, callback = train_complete_6stage_system(timesteps)
    
    # 最终性能评估
    evaluation_results = evaluate_final_performance(agent, env.unwrapped, callback.stage_manager)
    
    print(f"\n🎉 === 6阶段高级训练系统完成 === 🎉")
    
    # 判断训练是否成功
    overall_perf = evaluation_results['overall_performance']
    if overall_perf['stages_above_30_percent'] >= 5:
        print("🏆 训练非常成功！大部分阶段都达到了良好的性能！")
    elif overall_perf['stages_above_30_percent'] >= 3:
        print("✅ 训练基本成功！多数阶段表现良好，建议继续优化！")
    else:
        print("⚠️ 训练效果有限，建议调整参数或增加训练时间！")


if __name__ == '__main__':
    main()
