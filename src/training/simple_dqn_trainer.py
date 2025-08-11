"""
轻量化DQN训练器 - 专注于当前需求
只实现：DQN训练 + 固定用户位置 + 轨迹优化 + MRT/proportional波束
"""

import time
import numpy as np
from typing import Dict, Any, List
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

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
                        
                        if self.verbose > 0 and self.episode_count % 10 == 0:
                            print(f"Episode {self.episode_count}: reward={ep_info['r']:.2f}, length={ep_info['l']}")
        return True


class SimpleDQNTrainer:
    """
    轻量化DQN训练器
    专注于：固定用户位置 + 轨迹优化 + MRT/proportional波束
    """
    
    def __init__(self, 
                 num_users: int = 2,
                 num_antennas: int = 8,
                 transmit_power: float = 0.5,
                 beamforming_method: str = 'mrt',
                 power_strategy: str = 'proportional'):
        """
        初始化简化训练器
        
        Args:
            num_users: 用户数量
            num_antennas: 天线数量
            transmit_power: 发射功率
            beamforming_method: 波束成形方法 ('mrt', 'zf')
            power_strategy: 功率分配策略 ('equal', 'proportional')
        """
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.transmit_power = transmit_power
        self.beamforming_method = beamforming_method
        self.power_strategy = power_strategy
        
        # 训练组件
        self.env = None
        self.agent = None
        self.callback = None
        
    def create_environment(self) -> UAVEnvironment:
        """创建训练环境"""
        # 固定配置的环境
        env = UAVEnvironment(
            env_size=(100, 100, 50),
            num_users=self.num_users,
            num_antennas=self.num_antennas,
            start_position=(10, 10, 50),  # 固定起点
            end_position=(80, 80, 50),    # 固定终点
            flight_time=200.0,            # 减少到200秒，更快训练
            time_step=0.1,                # 0.1秒步长
            transmit_power=self.transmit_power,
            max_speed=30.0,
            min_speed=10.0,
            fixed_users=True,             # 固定用户位置
            seed=42                       # 固定随机种子
        )
        
        # 设置波束策略
        env.set_transmit_strategy(
            beamforming_method=self.beamforming_method,
            power_strategy=self.power_strategy
        )
        
        print(f"环境配置: {self.num_users}用户, {self.num_antennas}天线, {self.beamforming_method}+{self.power_strategy}")
        print(f"起点: {env.start_position}, 终点: {env.end_position}")
        print(f"用户位置: {env.get_user_positions()}")
        
        return env
    
    def setup_training(self, 
                      learning_rate: float = 1e-3,
                      gamma: float = 0.99,
                      batch_size: int = 32,
                      buffer_size: int = 50000,
                      exploration_fraction: float = 0.3,
                      exploration_final_eps: float = 0.05,
                      verbose: int = 1) -> None:
        """设置训练组件"""
        
        # 创建环境并包装
        base_env = self.create_environment()
        
        # 用Monitor包装来记录episode统计
        self.env = Monitor(base_env)
        
        # 创建DQN智能体
        self.agent = DQN(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=verbose,
            seed=42,
            # DQN特定参数
            learning_starts=1000,         # 开始学习前的步数
            train_freq=4,                 # 训练频率
            target_update_interval=1000,  # 目标网络更新间隔
        )
        
        # 创建回调
        self.callback = SimpleDQNCallback(verbose=verbose)
        
        print(f"DQN智能体配置完成")
        print(f"观测空间: {self.env.observation_space}")
        print(f"动作空间: {self.env.action_space}")
        
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """训练DQN智能体"""
        if self.agent is None:
            raise ValueError("请先调用 setup_training() 设置训练组件")
        
        print(f"开始DQN训练，总步数: {total_timesteps}")
        start_time = time.time()
        
        # 开始训练
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        # 收集统计信息
        stats = {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
            'total_episodes': self.callback.episode_count,
            'episode_rewards': self.callback.episode_rewards.copy(),
            'episode_lengths': self.callback.episode_lengths.copy(),
            'avg_reward': np.mean(self.callback.episode_rewards) if self.callback.episode_rewards else 0,
            'avg_length': np.mean(self.callback.episode_lengths) if self.callback.episode_lengths else 0,
        }
        
        print(f"训练完成!")
        print(f"  用时: {training_time:.1f}秒")
        print(f"  总episodes: {stats['total_episodes']}")
        print(f"  平均奖励: {stats['avg_reward']:.2f}")
        print(f"  平均长度: {stats['avg_length']:.1f}")
        
        return stats
    
    def evaluate_trajectory(self, deterministic: bool = True) -> Dict[str, Any]:
        """评估单条轨迹"""
        if self.agent is None:
            raise ValueError("智能体未训练")
        
        obs, _ = self.env.reset()
        trajectory = []
        rewards = []
        throughputs = []
        
        done = False
        step = 0
        
        while not done and step < 5000:  # 防止无限循环
            # 获取动作
            action, _ = self.agent.predict(obs, deterministic=deterministic)
            action = int(np.asarray(action).ravel()[0])  # 确保是int
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 记录信息
            trajectory.append(self.env.unwrapped.uav.get_position().copy())
            rewards.append(reward)
            throughputs.append(info.get('throughput', 0.0))
            
            step += 1
        
        trajectory = np.array(trajectory)
        
        return {
            'trajectory': trajectory,
            'rewards': rewards,
            'throughputs': throughputs,
            'total_reward': sum(rewards),
            'total_throughput': sum(throughputs),
            'steps': len(trajectory),
            'reached_end': terminated,
            'final_position': trajectory[-1] if len(trajectory) > 0 else None,
            'target_position': self.env.unwrapped.end_position
        }
    
    def evaluate_multiple_episodes(self, num_episodes: int = 10) -> List[Dict[str, Any]]:
        """评估多个episode"""
        results = []
        
        print(f"评估 {num_episodes} 个episodes...")
        
        for ep in range(num_episodes):
            result = self.evaluate_trajectory(deterministic=True)
            results.append(result)
            
            if ep % 5 == 0:
                print(f"Episode {ep}: 奖励={result['total_reward']:.2f}, "
                      f"步数={result['steps']}, 到达终点={result['reached_end']}")
        
        # 汇总统计
        total_rewards = [r['total_reward'] for r in results]
        episode_lengths = [r['steps'] for r in results]
        reached_end_count = sum(1 for r in results if r['reached_end'])
        
        print(f"评估完成:")
        print(f"  平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"  平均步数: {np.mean(episode_lengths):.1f}")
        print(f"  到达终点: {reached_end_count}/{num_episodes}")
        
        return results
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        if self.env is None:
            return {}
        
        env = self.env.unwrapped
        return {
            'start_position': env.start_position,
            'end_position': env.end_position,
            'user_positions': env.get_user_positions(),
            'env_size': env.env_size,
            'flight_time': env.flight_time,
            'time_step': env.time_step,
            'beamforming_method': env.default_beamforming_method,
            'power_strategy': env.default_power_strategy
        }
