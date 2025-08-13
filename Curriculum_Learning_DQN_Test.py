"""
课程学习DQN测试
从简单场景逐步增加复杂性，引导UAV学会正确的访问序列
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 确保src模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.uav_env import UAVEnvironment
from environment.curriculum_reward_config import (
    CurriculumRewardConfig, 
    CurriculumStageManager, 
    CurriculumRewardCalculator
)


class CurriculumLearningCallback(BaseCallback):
    """课程学习训练回调"""
    def __init__(self, stage_manager: CurriculumStageManager, verbose: int = 0):
        super().__init__(verbose)
        self.stage_manager = stage_manager
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.stage_history = []
        self.success_history = []

    def _on_step(self) -> bool:
        if len(self.locals.get('dones', [])) > 0:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and 'episode' in infos[i]:
                        ep_info = infos[i]['episode']
                        self.episode_rewards.append(float(ep_info['r']))
                        self.episode_lengths.append(int(ep_info['l']))
                        self.episode_count += 1
                        
                        # 课程学习评估
                        env = self.training_env.envs[0].unwrapped
                        
                        # 检查是否到达终点（距离判定）
                        uav_pos = env.uav.get_position()
                        end_pos = env.end_position
                        distance_to_end = np.linalg.norm(uav_pos - end_pos)
                        reached_end = distance_to_end <= env.reward_calculator.config.end_position_tolerance
                        
                        episode_result = {
                            'total_reward': float(ep_info['r']),
                            'reached_end': reached_end,
                            'users_visited': 0
                        }
                        
                        if hasattr(env, 'reward_calculator'):
                            stats = env.reward_calculator.get_stats()
                            episode_result['users_visited'] = stats.get('users_visited', 0)
                        
                        # 评估是否应该进入下一阶段
                        should_advance = self.stage_manager.evaluate_stage_performance(episode_result)
                        
                        # 记录历史
                        self.stage_history.append(self.stage_manager.current_stage)
                        self.success_history.append(episode_result['users_visited'] >= len(env.get_user_positions()) and episode_result['reached_end'])
                        
                        if should_advance:
                            self.stage_manager.advance_to_next_stage()
                            # 更新环境中的用户位置
                            self._update_environment_for_new_stage()
                        
                        if self.verbose > 0 and self.episode_count % 5 == 0:
                            stage_info = self.stage_manager.get_current_stage_info()
                            print(f"Episode {self.episode_count}: reward={ep_info['r']:.2f}, length={ep_info['l']}")
                            print(f"  阶段{stage_info['stage']}: 成功率={stage_info['success_rate']:.2%} ({stage_info['successes']}/{stage_info['episodes']})")
                            print(f"  用户访问: {episode_result['users_visited']}, 到达终点: {episode_result['reached_end']}")
        return True
    
    def _update_environment_for_new_stage(self):
        """为新阶段更新环境配置"""
        env = self.training_env.envs[0].unwrapped
        stage_config = self.stage_manager.get_stage_config(self.stage_manager.current_stage)
        
        # 更新用户位置
        new_user_positions = stage_config['user_positions']
        
        # 确保始终有2个用户位置
        if len(new_user_positions) == 1:
            # 阶段1：添加虚拟用户
            extended_positions = np.array([
                new_user_positions[0],  # 真实用户
                [200.0, 200.0, 0.0]     # 虚拟用户（很远）
            ])
        else:
            extended_positions = new_user_positions
        
        env.user_manager.set_user_positions(extended_positions)
        
        print(f"🔄 环境已更新至{stage_config['stage_name']}")
        print(f"   新用户位置: {new_user_positions[:, :2].tolist()}")
        if len(new_user_positions) == 1:
            print(f"   (添加虚拟用户位置: [200, 200])")


def create_curriculum_learning_environment():
    """创建课程学习UAV环境"""
    
    # 课程学习配置
    reward_config = CurriculumRewardConfig(
        # === 基础奖励 ===
        w_throughput_base=100.0,
        w_movement_bonus=15.0,
        
        # === 用户访问奖励 ===
        B_user_visit=2000.0,
        B_all_users_visited=3000.0,
        
        # === 终点奖励 ===
        B_reach_end=2000.0,
        B_mission_complete=5000.0,
        
        # === 引导奖励 ===
        w_user_approach=50.0,
        w_progress_bonus=30.0,
        
        # === 惩罚 ===
        w_stagnation=3.0,
        w_oob=100.0,
        
        # === 服务参数（调整为更宽松的条件）===
        user_service_radius=60.0,        # 增大服务半径：40->60m
        close_to_user_threshold=80.0,    # 增大接近阈值：60->80m  
        end_position_tolerance=25.0,     # 增大终点容忍：20->25m
        user_visit_time_threshold=0.8,   # 减少访问时间要求：1.5->0.8s
        
        # === 时间约束 ===
        min_flight_time=200.0,
        max_flight_time=300.0,
        time_step=0.1
    )
    
    # 创建阶段管理器
    stage_manager = CurriculumStageManager()
    
    env = UAVEnvironment(
        env_size=(100, 100, 50),
        num_users=2,  # 最大用户数，实际使用数由课程阶段决定
        num_antennas=8,
        start_position=(0, 0, 50),
        end_position=(80, 80, 50),
        flight_time=300.0,
        time_step=0.1,
        transmit_power=0.5,
        max_speed=30.0,
        min_speed=10.0,
        fixed_users=True,
        reward_config=None,
        seed=42
    )
    
    # 设置课程学习奖励计算器
    curriculum_calculator = CurriculumRewardCalculator(reward_config, stage_manager)
    env.set_reward_calculator(curriculum_calculator)
    
    # 设置beamforming策略
    env.set_transmit_strategy(
        beamforming_method='mrt',
        power_strategy='proportional'
    )
    
    # 初始化为阶段1的用户位置（需要扩展到2个用户）
    stage1_config = stage_manager.get_stage_config(1)
    stage1_positions = stage1_config['user_positions']
    
    # 对于阶段1（只有1个用户），需要添加一个虚拟用户位置
    if len(stage1_positions) == 1:
        # 添加一个远离的虚拟用户，不会影响训练
        extended_positions = np.array([
            stage1_positions[0],  # 真实用户
            [200.0, 200.0, 0.0]   # 虚拟用户（很远，不会被访问）
        ])
    else:
        extended_positions = stage1_positions
    
    env.user_manager.set_user_positions(extended_positions)
    
    return env, stage_manager


def train_curriculum_learning_dqn(env, stage_manager, total_timesteps=200000):
    """训练课程学习DQN"""
    monitored_env = Monitor(env)
    
    # DQN配置：适合课程学习
    agent = DQN(
        policy='MlpPolicy',
        env=monitored_env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=200000,
        exploration_fraction=0.6,  # 平衡探索和利用
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        seed=42,
        learning_starts=5000,
        train_freq=4,
        target_update_interval=2000,
        policy_kwargs=dict(
            net_arch=[256, 256, 128]
        )
    )
    
    callback = CurriculumLearningCallback(stage_manager, verbose=1)
    
    print("🎓 开始课程学习DQN训练...")
    print("📚 训练策略：从简单到复杂，逐步学习")
    print("🎯 阶段1：单用户 → 阶段2：双用户近距离 → 阶段3：双用户中距离 → 阶段4：完整场景")
    
    # 显示初始阶段信息
    stage1_config = stage_manager.get_stage_config(1)
    print(f"\n🚀 开始 {stage1_config['stage_name']}")
    print(f"   目标: {stage1_config['description']}")
    print(f"   用户位置: {stage1_config['user_positions'][:, :2].tolist()}")
    
    start_time = time.time()
    
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ 课程学习训练完成! 耗时: {training_time:.1f}秒")
    print(f"📈 总回合数: {callback.episode_count}")
    print(f"🎓 最终阶段: {stage_manager.current_stage}/4")
    
    if callback.episode_rewards:
        print(f"平均奖励: {np.mean(callback.episode_rewards):.2f}")
        print(f"最后10回合平均奖励: {np.mean(callback.episode_rewards[-10:]):.2f}")
        
        # 计算各阶段成功率
        final_success_rate = np.mean(callback.success_history[-20:]) if len(callback.success_history) >= 20 else 0
        print(f"最近20回合成功率: {final_success_rate:.2%}")
    
    return agent, callback, monitored_env


def evaluate_curriculum_trajectory(agent, env, stage_manager, deterministic=True):
    """评估课程学习轨迹（在最终阶段）"""
    
    # 确保在最终阶段（阶段4）评估
    stage4_config = stage_manager.get_stage_config(4)
    env.unwrapped.user_manager.set_user_positions(stage4_config['user_positions'])
    
    obs, _ = env.reset()
    
    trajectory = []
    rewards = []
    reward_breakdowns = []
    throughputs = []
    actions = []
    
    done = False
    step = 0
    
    print(f"🔍 在最终阶段评估课程学习效果...")
    print(f"   用户位置: {stage4_config['user_positions'][:, :2].tolist()}")
    
    while not done and step < 3000:
        action, _ = agent.predict(obs, deterministic=deterministic)
        action = int(np.asarray(action).ravel()[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        trajectory.append(env.unwrapped.uav.get_position().copy())
        rewards.append(reward)
        throughputs.append(info.get('throughput', 0.0))
        actions.append(action)
        
        # 记录奖励分解
        if hasattr(env.unwrapped, '_last_reward_breakdown'):
            reward_breakdowns.append(env.unwrapped._last_reward_breakdown.copy())
        else:
            reward_breakdowns.append({})
        
        step += 1
    
    trajectory = np.array(trajectory)
    
    return {
        'trajectory': trajectory,
        'rewards': rewards,
        'reward_breakdowns': reward_breakdowns,
        'throughputs': throughputs,
        'actions': actions,
        'total_reward': sum(rewards),
        'total_throughput': sum(throughputs),
        'steps': len(trajectory),
        'reached_end': terminated,
        'final_position': trajectory[-1] if len(trajectory) > 0 else None,
        'target_position': env.unwrapped.end_position
    }


def plot_curriculum_analysis(result, env, callback, stage_manager):
    """绘制课程学习分析图表"""
    
    plt.figure(figsize=(20, 16))
    
    # === 1. 训练收敛曲线（按阶段着色）===
    plt.subplot(4, 4, 1)
    episode_rewards = callback.episode_rewards
    stage_history = callback.stage_history
    
    # 按阶段着色绘制
    stage_colors = ['blue', 'green', 'orange', 'red']
    current_stage = 1
    start_idx = 0
    
    for i, stage in enumerate(stage_history):
        if stage != current_stage or i == len(stage_history) - 1:
            end_idx = i if stage != current_stage else i + 1
            if end_idx > start_idx:
                episodes = range(start_idx, end_idx)
                rewards = episode_rewards[start_idx:end_idx]
                plt.plot(episodes, rewards, color=stage_colors[current_stage-1], 
                        alpha=0.7, linewidth=1, label=f'阶段{current_stage}')
            current_stage = stage
            start_idx = i
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('📚 课程学习训练收敛（按阶段）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # === 2. 阶段转换历史 ===
    plt.subplot(4, 4, 2)
    episodes = range(len(stage_history))
    plt.plot(episodes, stage_history, 'b-', linewidth=2, marker='o', markersize=3)
    plt.ylim(0.5, 4.5)
    plt.yticks([1, 2, 3, 4], ['阶段1\n单用户', '阶段2\n双用户近', '阶段3\n双用户中', '阶段4\n完整场景'])
    plt.xlabel('Episode')
    plt.ylabel('课程阶段')
    plt.title('📈 课程阶段演进')
    plt.grid(True, alpha=0.3)
    
    # === 3. 成功率历史 ===
    plt.subplot(4, 4, 3)
    if callback.success_history:
        # 计算滑动成功率
        window_size = 10
        success_rates = []
        for i in range(len(callback.success_history)):
            start = max(0, i - window_size + 1)
            window_successes = callback.success_history[start:i+1]
            success_rates.append(np.mean(window_successes))
        
        episodes = range(len(success_rates))
        plt.plot(episodes, success_rates, 'g-', linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='目标成功率')
        plt.ylim(0, 1)
        plt.xlabel('Episode')
        plt.ylabel('成功率（滑动窗口）')
        plt.title('📊 学习成功率演进')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # === 4. 最终轨迹图 ===
    plt.subplot(4, 4, 4)
    trajectory = result['trajectory']
    if len(trajectory) > 0:
        # 绘制轨迹
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=3, alpha=0.8, label='最终评估轨迹')
        
        # 环境元素
        env_unwrapped = env.unwrapped
        plt.scatter(*env_unwrapped.start_position[:2], c='green', s=200, marker='o', 
                   label='起点', zorder=10, edgecolors='black', linewidth=2)
        plt.scatter(*env_unwrapped.end_position[:2], c='red', s=300, marker='*', 
                   label='终点', zorder=10, edgecolors='black', linewidth=2)
        
        # 最终阶段用户位置
        user_positions = env_unwrapped.get_user_positions()
        for i, user_pos in enumerate(user_positions):
            plt.scatter(user_pos[0], user_pos[1], c='purple', s=150, marker='x',
                       label=f'用户{i+1}' if i == 0 else "", zorder=8)
            
            # 服务圆圈
            service_circle = plt.Circle((user_pos[0], user_pos[1]), 40,
                                      fill=False, color='purple', linestyle='--',
                                      linewidth=2, alpha=0.6)
            plt.gca().add_patch(service_circle)
        
        # 终点容忍圆圈
        tolerance_circle = plt.Circle((80, 80), 20,
                                    fill=False, color='red', linestyle='-',
                                    linewidth=2, alpha=0.8)
        plt.gca().add_patch(tolerance_circle)
        
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('🎯 最终评估轨迹（阶段4）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # === 5. 奖励分解分析 ===
    plt.subplot(4, 4, 5)
    if result['reward_breakdowns']:
        approach_rewards = [rb.get('user_approach', 0) + rb.get('end_approach', 0) for rb in result['reward_breakdowns']]
        visit_bonuses = [rb.get('user_visit_bonus', 0) for rb in result['reward_breakdowns']]
        progress_bonuses = [rb.get('progress_bonus', 0) for rb in result['reward_breakdowns']]
        
        steps = range(len(approach_rewards))
        plt.plot(steps, approach_rewards, label='接近奖励', alpha=0.8)
        plt.plot(steps, visit_bonuses, label='访问奖励', alpha=0.8)
        plt.plot(steps, progress_bonuses, label='进展奖励', alpha=0.8)
        
        plt.xlabel('时间步')
        plt.ylabel('奖励值')
        plt.title('🎁 课程学习奖励分解')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # === 6. 累积奖励 ===
    plt.subplot(4, 4, 6)
    cumulative_rewards = np.cumsum(result['rewards'])
    plt.plot(cumulative_rewards, 'cyan', linewidth=2)
    plt.xlabel('时间步')
    plt.ylabel('累积奖励')
    plt.title('📈 最终评估累积奖励')
    plt.grid(True, alpha=0.3)
    
    # === 7. 动作分布 ===
    plt.subplot(4, 4, 7)
    actions = result['actions']
    action_names = ['东', '南', '西', '北', '悬停']
    action_counts = [actions.count(i) for i in range(5)]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    bars = plt.bar(action_names, action_counts, alpha=0.7, color=colors)
    plt.xlabel('动作类型')
    plt.ylabel('次数')
    plt.title('🎮 最终评估动作分布')
    plt.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + max(action_counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # === 8. 课程学习统计摘要 ===
    plt.subplot(4, 4, 8)
    plt.axis('off')
    
    # 分析最终结果
    final_stats = {}
    if hasattr(env.unwrapped, 'reward_calculator'):
        final_stats = env.unwrapped.reward_calculator.get_stats()
    
    visited_users = final_stats.get('user_visited_flags', [])
    visit_order = final_stats.get('user_visit_order', [])
    users_visited = final_stats.get('users_visited', 0)
    
    final_pos = result['final_position']
    target_pos = result['target_position']
    final_distance = np.linalg.norm(final_pos - target_pos) if final_pos is not None else float('inf')
    
    # 计算整体成功率
    total_success_rate = np.mean(callback.success_history) if callback.success_history else 0
    final_success_rate = np.mean(callback.success_history[-20:]) if len(callback.success_history) >= 20 else 0
    
    summary_text = f"""
📚 课程学习训练报告

🎓 训练统计:
• 总回合数: {callback.episode_count}
• 最终阶段: {stage_manager.current_stage}/4
• 整体成功率: {total_success_rate:.2%}
• 最终成功率: {final_success_rate:.2%}

🎯 最终评估结果:
• 访问用户: {visited_users}
• 访问顺序: {visit_order}
• 到达终点: {'是' if result['reached_end'] else '否'}
• 最终距离: {final_distance:.1f}m
• 总奖励: {result['total_reward']:.0f}

🏆 课程学习效果:
{
'🎉 完美成功！完全掌握任务!' if users_visited == 2 and result['reached_end']
else f'🔶 部分成功：访问{users_visited}/2用户' + ('，到达终点' if result['reached_end'] else '，未到终点')
if users_visited > 0 or result['reached_end']
else '❌ 课程学习需要调整'
}

💡 学习质量:
{
'🌟 课程学习策略有效!' if final_success_rate > 0.3
else '📚 需要更多训练或调整课程'
}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def main():
    print("📚 === 课程学习DQN训练 - 从简单到复杂 === 📚")
    print("策略：逐步增加任务复杂性，引导UAV学会正确的访问序列")
    print("目标：通过渐进式学习克服直接训练的困难\n")
    
    # 1. 创建课程学习环境
    env, stage_manager = create_curriculum_learning_environment()
    print("✅ 课程学习环境创建完成")
    
    # 2. 训练课程学习DQN
    agent, callback, monitored_env = train_curriculum_learning_dqn(
        env, stage_manager, total_timesteps=200000
    )
    
    # 3. 评估最终效果
    print("\n📊 在最终阶段评估课程学习效果...")
    result = evaluate_curriculum_trajectory(agent, monitored_env, stage_manager, deterministic=True)
    
    print(f"\n📚 === 课程学习最终评估结果 === 📚")
    print(f"总奖励: {result['total_reward']:.2f}")
    print(f"总吞吐量: {result['total_throughput']:.2f}")
    print(f"步数: {result['steps']}")
    print(f"到达终点: {result['reached_end']}")
    
    if result['final_position'] is not None:
        distance_to_target = np.linalg.norm(result['final_position'] - result['target_position'])
        print(f"最终距离: {distance_to_target:.2f}m")
        print(f"容忍度: 20.0m")
        
        # 分析课程学习效果
        if hasattr(monitored_env.unwrapped, 'reward_calculator'):
            final_stats = monitored_env.unwrapped.reward_calculator.get_stats()
            visited_users = final_stats.get('user_visited_flags', [])
            visit_order = final_stats.get('user_visit_order', [])
            users_visited = final_stats.get('users_visited', 0)
            
            print(f"访问用户: {visited_users}")
            print(f"访问顺序: {visit_order}")
            print(f"访问完成度: {users_visited}/2")
            
            # 最终评估
            if users_visited == 2 and result['reached_end']:
                print("🎉 课程学习完全成功！UAV学会了完整的任务序列！")
                print("✨ 从简单到复杂的学习策略证明有效！")
            elif users_visited >= 1:
                print(f"🔶 课程学习部分成功：学会了访问{users_visited}个用户")
                print("💪 说明课程学习策略开始奏效，需要进一步调优")
            else:
                print("❌ 课程学习效果有限，可能需要调整课程设计")
                print("💡 建议：简化初始阶段或增加更多渐进步骤")
            
            # 整体成功率分析
            final_success_rate = np.mean(callback.success_history[-20:]) if len(callback.success_history) >= 20 else 0
            print(f"最近20回合成功率: {final_success_rate:.2%}")
            
            if final_success_rate > 0.5:
                print("🌟 课程学习显著提升了任务完成能力！")
            elif final_success_rate > 0.2:
                print("📈 课程学习有一定效果，但还有提升空间")
            else:
                print("📚 课程学习需要进一步优化设计")
    
    # 4. 绘制课程学习分析
    plot_curriculum_analysis(result, monitored_env, callback, stage_manager)
    
    print(f"\n📚 === 课程学习DQN训练完成 === 📚")
    print("🎯 课程学习通过逐步增加复杂性，为解决序列访问问题提供了新思路")


if __name__ == '__main__':
    main()
