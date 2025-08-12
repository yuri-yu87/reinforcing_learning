"""
独立DQN测试脚本 - 避免复杂导入问题
专注于：DQN训练 + 固定用户位置 + 轨迹优化 + MRT/proportional波束
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 确保能找到src模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# 导入必要模块
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# 直接导入环境
from environment.uav_env import UAVEnvironment


class SimpleDQNCallback(BaseCallback):
    """简化的DQN回调，只收集必要统计信息"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """每步调用，收集episode结束时的统计"""
        # 检查是否有episode结束
        if len(self.locals.get('dones', [])) > 0:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    # 从info中获取episode统计
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and 'episode' in infos[i]:
                        ep_info = infos[i]['episode']
                        self.episode_rewards.append(float(ep_info['r']))
                        self.episode_lengths.append(int(ep_info['l']))
                        self.episode_count += 1
                        
                        if self.verbose > 0 and self.episode_count % 5 == 0:
                            print(f"Episode {self.episode_count}: reward={ep_info['r']:.2f}, length={ep_info['l']}")
        return True


def create_simple_environment():
    """创建符合新5-way离散策略的UAV环境"""
    from environment.reward_config import RewardConfig
    
    # 简化修复版奖励配置 - 确保系统正常工作
    reward_config = RewardConfig(
        # === 简化核心吞吐量奖励 ===
        w_throughput_base=100.0,           # 简化基础吞吐量权重
        w_throughput_multiplier=0.0,       # 禁用复杂距离调制
        
        # === 简化移动激励 ===
        w_movement_bonus=10.0,             # 简化移动奖励
        w_distance_progress=0.0,           # 禁用距离进展 (避免问题)
        w_user_approach=50.0,              # 简化用户接近奖励
        
        # === 降低惩罚确保稳定性 ===
        w_oob=100.0,                       # 适中出界惩罚
        w_stagnation=1.0,                  # 最小停滞惩罚
        
        # === 简化终端奖励 ===
        B_mission_complete=1000.0,         # 清晰任务完成信号
        B_reach_end=500.0,                # 清晰终点到达奖励
        B_time_window=500.0,              # 清晰时间窗口奖励
        B_fair_access=200.0,              # 清晰公平访问奖励
        B_visit_all_users=300.0,          # 清晰访问用户奖励
        
        # === 关键修复 ===
        alpha_fair=0.0,                   # 禁用proportional fair (关键修复!)
        user_service_radius=50.0,          # 扩大服务半径 (vs 10.0)
        
        # === 放宽距离阈值 ===
        close_to_user_threshold=40.0,      # 放宽用户接近阈值
        close_to_end_threshold=30.0,       # 放宽终点接近阈值
        
        # === 时间约束 ===
        min_flight_time=200.0,
        max_flight_time=300.0,
        
        # === 放宽任务完成参数 ===
        end_position_tolerance=20.0,       # 大幅放宽终点容忍度 (vs 10.0)
        user_visit_time_threshold=0.5,     # 降低用户访问时间要求 (vs 1.0)
        
        # === 放宽停滞检测 ===
        stagnation_threshold=1.0,          # 更严格检测 (vs 2.0)
        stagnation_time_window=3.0,        # 更短窗口 (vs 5.0)
        
        # === 系统参数 ===
        time_step=0.1
    )
    
    env = UAVEnvironment(
        env_size=(100, 100, 50),
        num_users=2,
        num_antennas=8,
        start_position=(0, 0, 50),    # 起点
        end_position=(80, 80, 50),    # 终点
        flight_time=300.0,            # 匹配时间窗口上限
        time_step=0.1,                # 0.1秒步长
        transmit_power=0.5,
        max_speed=30.0,
        min_speed=10.0,
        fixed_users=True,             # 固定用户位置
        reward_config=reward_config,  # 使用新的离散5-way策略奖励配置
        seed=42                       # 固定随机种子
    )
    
    # 设置波束策略
    env.set_transmit_strategy(
        beamforming_method='mrt',
        power_strategy='proportional'
    )
    
    print(f"环境配置: 2用户, 8天线, mrt+proportional")
    print(f"起点: {env.start_position}, 终点: {env.end_position}")
    print(f"简化修复策略: 离散5-way RL + 简化奖励 + 确保稳定性")
    
    # 手动设置用户位置来确保有用户
    fixed_positions = np.array([
        [15.0, 75.0, 0.0],   
        [75.0, 15.0, 0.0]    
    ])
    env.user_manager.set_user_positions(fixed_positions)
    
    print(f"用户位置: {env.get_user_positions()}")
    print(f"飞行时间: {env.flight_time}s, 步长: {env.time_step}s, 预期步数: {int(env.flight_time/env.time_step)}")
    print(f"关键修复: 服务半径={reward_config.user_service_radius}m, 禁用proportional fair={reward_config.alpha_fair}")
    print(f"奖励权重: 吞吐量基础={reward_config.w_throughput_base}, 移动={reward_config.w_movement_bonus}")
    print(f"终端奖励: 完整任务={reward_config.B_mission_complete}, 到达终点={reward_config.B_reach_end}")
    print(f"时间约束: [{reward_config.min_flight_time}, {reward_config.max_flight_time}]s")
    
    return env


def train_simple_dqn(env, total_timesteps=80000):
    """训练新策略DQN - 针对离散5-way约束RL优化"""
    # 用Monitor包装来记录episode统计
    monitored_env = Monitor(env)
    
    # 创建DQN智能体 - 针对新策略优化
    agent = DQN(
        policy='MlpPolicy',
        env=monitored_env,
        learning_rate=5e-4,           # 适中学习率，稳定学习
        gamma=0.995,                  # 高折扣因子，重视长期回报
        batch_size=64,                # 较大批次，稳定梯度
        buffer_size=200000,           # 大缓冲区，丰富经验
        exploration_fraction=0.7,     # 70%时间探索，充分学习环境
        exploration_initial_eps=1.0,  # 初始完全随机
        exploration_final_eps=0.02,   # 最终少量随机
        verbose=1,
        seed=42,
        # DQN特定参数  
        learning_starts=2000,         # 充分收集经验后开始学习
        train_freq=8,                 # 适中训练频率
        target_update_interval=2000,  # 稳定的目标网络更新
        gradient_steps=1,             # 每次更新1步
        tau=1.0,                      # 硬更新目标网络
        policy_kwargs=dict(
            net_arch=[256, 256, 128]  # 更深网络，适应复杂奖励
        )
    )
    
    # 创建回调
    callback = SimpleDQNCallback(verbose=1)
    
    print(f"DQN智能体配置完成")
    print(f"观测空间: {monitored_env.observation_space}")
    print(f"动作空间: {monitored_env.action_space}")
    print(f"网络架构: {agent.policy_kwargs}")
    print(f"超参数: lr={agent.learning_rate}, gamma={agent.gamma}, batch_size={agent.batch_size}")
    print(f"探索策略: 初始ε={agent.exploration_initial_eps}, 最终ε={agent.exploration_final_eps}, 探索比例={agent.exploration_fraction}")
    
    print(f"\n开始新策略DQN训练，总步数: {total_timesteps}")
    start_time = time.time()
    
    # 开始训练
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    
    print(f"训练完成! 用时: {training_time:.1f}秒")
    print(f"总episodes: {callback.episode_count}")
    if callback.episode_rewards:
        print(f"平均奖励: {np.mean(callback.episode_rewards):.2f}")
        print(f"最终10个episode平均奖励: {np.mean(callback.episode_rewards[-10:]):.2f}")
    
    return agent, callback, monitored_env


def evaluate_trajectory(agent, env, deterministic=True):
    """评估单条轨迹"""
    obs, _ = env.reset()
    
    # 重新设置用户位置确保一致
    fixed_positions = np.array([
        [15.0, 75.0, 0.0],   
        [75.0, 15.0, 0.0]    
    ])
    env.unwrapped.user_manager.set_user_positions(fixed_positions)
    
    trajectory = []
    rewards = []
    throughputs = []
    actions = []
    focus_users = []  # 跟踪专注用户变化
    cumulative_services = []  # 跟踪累积服务
    
    done = False
    step = 0
    
    while not done and step < 3000:  # 防止无限循环
        # 获取动作
        action, _ = agent.predict(obs, deterministic=deterministic)
        action = int(np.asarray(action).ravel()[0])  # 确保是int
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 记录信息
        trajectory.append(env.unwrapped.uav.get_position().copy())
        rewards.append(reward)
        throughputs.append(info.get('throughput', 0.0))
        actions.append(action)
        
        # 记录专注机制信息（如果有奖励详情）
        if hasattr(env.unwrapped, '_last_reward_breakdown') and env.unwrapped._last_reward_breakdown:
            rb = env.unwrapped._last_reward_breakdown
            focus_users.append(rb.get('current_focus_user', -1))
            cumulative_services.append(dict(rb.get('cumulative_services', {})))
        else:
            focus_users.append(-1)
            cumulative_services.append({})
        
        step += 1
    
    trajectory = np.array(trajectory)
    
    return {
        'trajectory': trajectory,
        'rewards': rewards,
        'throughputs': throughputs,
        'actions': actions,
        'focus_users': focus_users,
        'cumulative_services': cumulative_services,
        'total_reward': sum(rewards),
        'total_throughput': sum(throughputs),
        'steps': len(trajectory),
        'reached_end': terminated,
        'final_position': trajectory[-1] if len(trajectory) > 0 else None,
        'target_position': env.unwrapped.end_position
    }


def plot_training_results(callback):
    """绘制训练结果"""
    episode_rewards = callback.episode_rewards
    episode_lengths = callback.episode_lengths
    
    if len(episode_rewards) == 0:
        print("没有收集到episode数据")
        return
    
    plt.figure(figsize=(15, 5))
    
    # 原始奖励
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('训练收敛曲线')
    plt.grid(True, alpha=0.3)
    
    # 滑动平均
    plt.subplot(1, 3, 2)
    window_size = min(10, len(episode_rewards) // 4) 
    if window_size > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed, color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title(f'滑动平均 (窗口={window_size})')
        plt.grid(True, alpha=0.3)
    
    # episode长度
    plt.subplot(1, 3, 3)
    plt.plot(episode_lengths, alpha=0.6, color='green', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode长度')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"总Episodes: {len(episode_rewards)}")
    print(f"最终10个episode平均奖励: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"最终10个episode平均长度: {np.mean(episode_lengths[-10:]):.1f}")


def plot_trajectory_analysis(result, env):
    """绘制轨迹分析"""
    trajectory = result['trajectory']
    
    if len(trajectory) == 0:
        print("没有轨迹数据")
        return
    
    env_unwrapped = env.unwrapped
    
    plt.figure(figsize=(16, 12))
    
    # 1. 轨迹图
    plt.subplot(3, 3, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='UAV轨迹')
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=100, marker='o', label='起点')
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=100, marker='*', label='终点')
    
    # 用户位置
    user_positions = env_unwrapped.get_user_positions()
    for i, user_pos in enumerate(user_positions):
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=80, marker='x', 
                   label=f'用户{i+1}' if i == 0 else "")
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('优化轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # 2. 奖励曲线
    plt.subplot(3, 3, 2)
    plt.plot(result['rewards'], 'g-', linewidth=1.5)
    plt.xlabel('步数')
    plt.ylabel('即时奖励')
    plt.title('步奖励曲线')
    plt.grid(True, alpha=0.3)
    
    # 3. 吞吐曲线
    plt.subplot(3, 3, 3)
    plt.plot(result['throughputs'], 'orange', linewidth=1.5)
    plt.xlabel('步数')
    plt.ylabel('吞吐量')
    plt.title('步吞吐曲线')
    plt.grid(True, alpha=0.3)
    
    # 4. 到目标距离
    plt.subplot(3, 3, 4)
    target_pos = env_unwrapped.end_position
    distances = [np.linalg.norm(pos[:2] - target_pos[:2]) for pos in trajectory]
    plt.plot(distances, 'm-', linewidth=1.5)
    plt.xlabel('步数')
    plt.ylabel('到终点距离 (m)')
    plt.title('到终点距离')
    plt.grid(True, alpha=0.3)
    
    # 5. 累积奖励
    plt.subplot(3, 3, 5)
    cumulative_rewards = np.cumsum(result['rewards'])
    plt.plot(cumulative_rewards, 'cyan', linewidth=1.5)
    plt.xlabel('步数')
    plt.ylabel('累积奖励')
    plt.title('累积奖励')
    plt.grid(True, alpha=0.3)
    
    # 6. 累积吞吐
    plt.subplot(3, 3, 6)
    cumulative_throughput = np.cumsum(result['throughputs'])
    plt.plot(cumulative_throughput, 'brown', linewidth=1.5)
    plt.xlabel('步数')
    plt.ylabel('累积吞吐')
    plt.title('累积吞吐')
    plt.grid(True, alpha=0.3)
    
    # 7. 动作分布
    plt.subplot(3, 3, 7)
    actions = result['actions']
    action_names = ['East', 'South', 'West', 'North', 'Hover']
    action_counts = [actions.count(i) for i in range(5)]
    plt.bar(action_names, action_counts, alpha=0.7, color=['red', 'blue', 'green', 'orange', 'purple'])
    plt.xlabel('动作类型')
    plt.ylabel('次数')
    plt.title('动作分布')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 8. 轨迹热力图
    plt.subplot(3, 3, 8)
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    plt.hist2d(x_coords, y_coords, bins=20, alpha=0.6, cmap='YlOrRd')
    plt.colorbar(label='停留时间')
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=100, marker='o')
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=100, marker='*')
    for user_pos in user_positions:
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=80, marker='x')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('轨迹热力图')
    plt.gca().set_aspect('equal')
    
    # 9. 用户专注变化（如果有数据）
    plt.subplot(3, 3, 9)
    if 'focus_users' in result and len(result['focus_users']) > 0:
        focus_data = result['focus_users']
        steps = range(len(focus_data))
        plt.plot(steps, focus_data, 'purple', linewidth=2, marker='o', markersize=2)
        plt.xlabel('步数')
        plt.ylabel('专注用户ID')
        plt.title('用户专注变化')
        plt.yticks([0, 1], ['用户0', '用户1'])
        plt.grid(True, alpha=0.3)
    else:
        # 备用：速度曲线
        if len(trajectory) > 1:
            speeds = []
            for i in range(1, len(trajectory)):
                displacement = np.linalg.norm(trajectory[i] - trajectory[i-1])
                speed = displacement / env_unwrapped.time_step
                speeds.append(speed)
            plt.plot(speeds, 'navy', linewidth=1.5)
            plt.xlabel('步数')
            plt.ylabel('速度 (m/s)')
            plt.title('速度曲线')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    print("=== 新策略DQN训练测试 - 离散5-way约束RL ===")
    
    # 1. 创建环境
    env = create_simple_environment()
    
    # 2. 训练DQN
    agent, callback, monitored_env = train_simple_dqn(env, total_timesteps=80000)
    
    # 3. 绘制训练结果
    print("\n绘制训练收敛曲线...")
    plot_training_results(callback)
    
    # 4. 评估单条轨迹
    print("\n评估单条轨迹...")
    result = evaluate_trajectory(agent, monitored_env, deterministic=True)
    
    print("单条轨迹评估:")
    print(f"  总奖励: {result['total_reward']:.2f}")
    print(f"  总吞吐: {result['total_throughput']:.2f}")
    print(f"  步数: {result['steps']}")
    print(f"  到达终点: {result['reached_end']}")
    print(f"  最终位置: {result['final_position']}")
    print(f"  目标位置: {result['target_position']}")
    
    if result['final_position'] is not None:
        distance_to_target = np.linalg.norm(result['final_position'] - result['target_position'])
        print(f"  到目标距离: {distance_to_target:.2f}m")
    
    # 5. 绘制详细轨迹分析
    print("\n绘制轨迹分析...")
    plot_trajectory_analysis(result, monitored_env)
    
    # 6. 评估多条轨迹
    print("\n评估多条轨迹...")
    results = []
    for ep in range(5):
        result = evaluate_trajectory(agent, monitored_env, deterministic=True)
        results.append(result)
        print(f"Episode {ep}: 奖励={result['total_reward']:.2f}, 步数={result['steps']}, 到达终点={result['reached_end']}")
    
    # 绘制多条轨迹对比
    plt.figure(figsize=(12, 5))
    
    # 所有轨迹
    plt.subplot(1, 2, 1)
    for i, result in enumerate(results):
        trajectory = result['trajectory']
        if len(trajectory) > 0:
            alpha = 0.8 if result['reached_end'] else 0.4
            color = 'blue' if result['reached_end'] else 'red'
            plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha, linewidth=2)
    
    # 标记起点、终点、用户
    env_unwrapped = monitored_env.unwrapped
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=100, marker='o', label='起点')
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=150, marker='*', label='终点')
    user_positions = env_unwrapped.get_user_positions()
    for i, user_pos in enumerate(user_positions):
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=80, marker='x', 
                   label=f'用户{i+1}' if i == 0 else "")
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('5条轨迹 (蓝=成功到达, 红=未到达)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # 性能统计
    plt.subplot(1, 2, 2)
    rewards = [r['total_reward'] for r in results]
    reached_flags = [r['reached_end'] for r in results]
    
    colors = ['green' if flag else 'red' for flag in reached_flags]
    plt.bar(range(len(rewards)), rewards, color=colors, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('总奖励')
    plt.title('各Episode性能')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 成功率统计
    success_rate = sum(reached_flags) / len(reached_flags) * 100
    print(f"\n成功到达终点: {sum(reached_flags)}/{len(reached_flags)} ({success_rate:.1f}%)")
    
    print("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
