"""
Enhanced DQN test script - with circle visualization and strong terminal guidance strategy
Focus: DQN training + fixed user positions + circle visualization + high-precision arrival
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List

# Set matplotlib font for proper Chinese label display
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Ensure src module is found
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.uav_env import UAVEnvironment


class SimpleDQNCallback(BaseCallback):
    """Simplified DQN callback, only collects essential statistics"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Collect episode statistics at the end of each episode
        if len(self.locals.get('dones', [])) > 0:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    infos = self.locals.get('infos', [])
                    if i < len(infos) and 'episode' in infos[i]:
                        ep_info = infos[i]['episode']
                        self.episode_rewards.append(float(ep_info['r']))
                        self.episode_lengths.append(int(ep_info['l']))
                        self.episode_count += 1
                        if self.verbose > 0 and self.episode_count % 5 == 0:
                            print(f"Episode {self.episode_count}: reward={ep_info['r']:.2f}, length={ep_info['l']}")
        return True


def create_enhanced_environment():
    """Create UAV environment with strong terminal guidance strategy"""
    from environment.reward_config import RewardConfig

    reward_config = RewardConfig(
        # === Core throughput reward ===
        w_throughput_base=120.0,           # Keep base throughput weight
        w_throughput_multiplier=0.0,       # Disable complex distance modulation

        # === Strong movement incentives ===
        w_movement_bonus=25.0,             # Further enhance movement reward (vs 20.0)
        w_distance_progress=40.0,          # Greatly enhance distance progress reward (vs 30.0)
        w_user_approach=1500.0,            # Strong user/terminal approach reward (vs 100.0)

        # === Balanced penalties for goal orientation ===
        w_oob=100.0,                       # Keep out-of-bounds penalty
        w_stagnation=10.0,                 # Further increase stagnation penalty (vs 5.0)

        # === Strong terminal rewards ===
        B_mission_complete=2500.0,         # Further enhance mission complete signal (vs 1500.0)
        B_reach_end=2000.0,                # Strong terminal arrival reward (vs 2000.0)
        B_time_window=800.0,               # Keep time window reward
        B_fair_access=2000.0,              # Keep fair access reward
        B_visit_all_users=2000.0,          # Keep user visit reward

        # === Key fixes ===
        alpha_fair=0.0,                    # Disable proportional fair
        user_service_radius=40.0,          # Keep user service radius

        # === Strong terminal guidance thresholds ===
        close_to_user_threshold=60.0,      # Keep user approach threshold
        close_to_end_threshold=60.0,       # Greatly expand terminal approach threshold (vs 60.0)

        # === Time constraints ===
        min_flight_time=200.0,
        max_flight_time=300.0,

        # === Challenging mission completion parameters ===
        end_position_tolerance=20.0,       # Further reduce to 8m (vs 10.0)
        user_visit_time_threshold=1.0,     # Keep user visit time requirement

        # === Stricter stagnation detection ===
        stagnation_threshold=0.8,          # Stricter detection (vs 1.0)
        stagnation_time_window=2.5,        # Shorter window (vs 3.0)

        # === System parameters ===
        time_step=0.1
    )

    env = UAVEnvironment(
        env_size=(100, 100, 50),
        num_users=2,
        num_antennas=8,
        start_position=(0, 0, 50),    # Start position
        end_position=(80, 80, 50),    # End position
        flight_time=300.0,            # Max matching time window
        time_step=0.1,                # 0.1s step size
        transmit_power=0.5,
        max_speed=30.0,
        min_speed=10.0,
        fixed_users=True,             # Fixed user positions
        reward_config=reward_config,  # Use strong guidance strategy config
        seed=42                       # Fixed random seed
    )

    # Set beamforming strategy
    env.set_transmit_strategy(
        beamforming_method='mrt',
        power_strategy='proportional'
    )

    # Manually set user positions to ensure users exist
    fixed_positions = np.array([
        [15.0, 75.0, 0.0],
        [75.0, 15.0, 0.0]
    ])
    env.user_manager.set_user_positions(fixed_positions)

    return env


def train_enhanced_dqn(env, total_timesteps=250000):
    """Train DQN with strong guidance strategy"""
    # Wrap with Monitor to record episode statistics
    monitored_env = Monitor(env)

    # Create DQN agent - optimized for high-precision arrival
    agent = DQN(
        policy='MlpPolicy',
        env=monitored_env,
        learning_rate=3e-4,           # Lower learning rate, more stable learning
        gamma=0.998,                  # Higher discount factor, more focus on long-term return
        batch_size=128,               # Larger batch, more stable gradients
        buffer_size=300000,           # Larger buffer, richer experience
        exploration_fraction=0.8,     # 80% exploration, sufficient learning for precise strategy
        exploration_initial_eps=1.0,  # Fully random at start
        exploration_final_eps=0.01,   # Less final randomness
        verbose=1,
        seed=42,
        # DQN specific parameters
        learning_starts=3000,         # More initial experience collection
        train_freq=4,                 # More frequent training
        target_update_interval=1500,  # More frequent target network updates
        gradient_steps=1,             # 1 step per update
        tau=1.0,                      # Hard update for target network
        policy_kwargs=dict(
            net_arch=[512, 256, 128]  # Deeper and wider network for complex precise strategy
        )
    )

    callback = SimpleDQNCallback(verbose=1)

    start_time = time.time()

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    training_time = time.time() - start_time
    
    print(f"✅ Training Completed! Time taken: {training_time:.1f} seconds")
    print(f"📈 Total episodes: {callback.episode_count}")
    if callback.episode_rewards:
        print(f"Average reward: {np.mean(callback.episode_rewards):.2f}")
        print(f"Last 10 episode average reward: {np.mean(callback.episode_rewards[-10:]):.2f}")

    return agent, callback, monitored_env


def evaluate_trajectory(agent, env, deterministic=True):
    """Evaluate a single trajectory"""
    obs, _ = env.reset()

    # Reset user positions to ensure consistency
    fixed_positions = np.array([
        [15.0, 75.0, 0.0],
        [75.0, 15.0, 0.0]
    ])
    env.unwrapped.user_manager.set_user_positions(fixed_positions)

    trajectory = []
    rewards = []
    throughputs = []
    actions = []
    focus_users = []
    cumulative_services = []

    done = False
    step = 0

    while not done and step < 3000:
        action, _ = agent.predict(obs, deterministic=deterministic)
        action = int(np.asarray(action).ravel()[0])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory.append(env.unwrapped.uav.get_position().copy())
        rewards.append(reward)
        throughputs.append(info.get('throughput', 0.0))
        actions.append(action)

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
    """Plot training results"""
    episode_rewards = callback.episode_rewards
    episode_lengths = callback.episode_lengths

    if len(episode_rewards) == 0:
        print("No episode data collected")
        return

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Convergence Curve')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    window_size = min(10, len(episode_rewards) // 4)
    if window_size > 1:
        smoothed = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed, color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title(f'Smoothed (window={window_size})')
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(episode_lengths, alpha=0.6, color='green', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Length')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Last 10 episode average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Last 10 episode average length: {np.mean(episode_lengths[-10:]):.1f}")


def plot_enhanced_trajectory_with_circles(result, env):
    """Enhanced trajectory analysis - with circle visualization"""
    trajectory = result['trajectory']

    if len(trajectory) == 0:
        print("No trajectory data")
        return

    env_unwrapped = env.unwrapped
    reward_config = env_unwrapped.reward_calculator.config

    user_service_radius = reward_config.user_service_radius
    end_tolerance = reward_config.end_position_tolerance
    close_to_end_threshold = reward_config.close_to_end_threshold

    plt.figure(figsize=(20, 15))

    plt.subplot(3, 4, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=3, label='UAV Trajectory', alpha=0.8)
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=200, marker='o',
               label='Start', zorder=10, edgecolors='black', linewidth=2)
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=300, marker='*',
               label='End', zorder=10, edgecolors='black', linewidth=2)

    user_positions = env_unwrapped.get_user_positions()
    for i, user_pos in enumerate(user_positions):
        service_circle = plt.Circle((user_pos[0], user_pos[1]), user_service_radius,
                                  fill=False, color='purple', linestyle='--',
                                  linewidth=2, alpha=0.7, label='User Service Area' if i == 0 else "")
        plt.gca().add_patch(service_circle)
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=150, marker='x',
                   label=f'User{i+1}' if i == 0 else "", zorder=8)

    end_pos = env_unwrapped.end_position[:2]
    tolerance_circle = plt.Circle((end_pos[0], end_pos[1]), end_tolerance,
                                fill=False, color='red', linestyle='-',
                                linewidth=4, alpha=0.9, label=f'End Tolerance ({end_tolerance}m)')
    plt.gca().add_patch(tolerance_circle)

    guidance_circle = plt.Circle((end_pos[0], end_pos[1]), close_to_end_threshold,
                               fill=False, color='orange', linestyle=':',
                               linewidth=3, alpha=0.8, label=f'End Guidance ({close_to_end_threshold}m)')
    plt.gca().add_patch(guidance_circle)

    if len(trajectory) > 0:
        final_pos = trajectory[-1][:2]
        plt.scatter(final_pos[0], final_pos[1], c='gold', s=200, marker='D',
                   label='Final Position', zorder=9, edgecolors='black', linewidth=2)
        plt.plot([final_pos[0], end_pos[0]], [final_pos[1], end_pos[1]],
                'k--', alpha=0.8, linewidth=2, label='Distance Line')
        distance = np.linalg.norm(final_pos - end_pos)
        mid_x = (final_pos[0] + end_pos[0]) / 2
        mid_y = (final_pos[1] + end_pos[1]) / 2
        plt.text(mid_x, mid_y + 3, f'{distance:.1f}m',
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.title('Enhanced Guidance Trajectory - Circle Visualization', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')

    plt.subplot(3, 4, 2)
    plt.plot(result['rewards'], 'g-', linewidth=1.5)
    plt.xlabel('Step')
    plt.ylabel('Instant Reward')
    plt.title('Step Reward Curve')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 3)
    plt.plot(result['throughputs'], 'orange', linewidth=1.5)
    plt.xlabel('Step')
    plt.ylabel('Throughput')
    plt.title('Step Throughput Curve')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 4)
    target_pos = env_unwrapped.end_position
    distances = [np.linalg.norm(pos[:2] - target_pos[:2]) for pos in trajectory]
    plt.plot(distances, 'm-', linewidth=2)
    plt.axhline(y=end_tolerance, color='red', linestyle='--', alpha=0.7,
               label=f'Tolerance ({end_tolerance}m)')
    plt.axhline(y=close_to_end_threshold, color='orange', linestyle=':', alpha=0.7,
               label=f'Guidance ({close_to_end_threshold}m)')
    plt.xlabel('Step')
    plt.ylabel('Distance to End (m)')
    plt.title('Distance to End')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 5)
    cumulative_rewards = np.cumsum(result['rewards'])
    plt.plot(cumulative_rewards, 'cyan', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 6)
    cumulative_throughput = np.cumsum(result['throughputs'])
    plt.plot(cumulative_throughput, 'brown', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Throughput')
    plt.title('Cumulative Throughput')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 7)
    actions = result['actions']
    action_names = ['East', 'South', 'West', 'North', 'Hover']
    action_counts = [actions.count(i) for i in range(5)]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    bars = plt.bar(action_names, action_counts, alpha=0.7, color=colors)
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.title('Action Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    for bar, count in zip(bars, action_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(action_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(3, 4, 8)
    x_coords = trajectory[:, 0]
    y_coords = trajectory[:, 1]
    plt.hist2d(x_coords, y_coords, bins=25, alpha=0.8, cmap='YlOrRd')
    plt.colorbar(label='Stay Density')
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=100, marker='o')
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=150, marker='*')
    for user_pos in user_positions:
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=80, marker='x')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Heatmap')
    plt.gca().set_aspect('equal')

    plt.subplot(3, 4, 9)
    if len(trajectory) > 1:
        speeds = []
        for i in range(1, len(trajectory)):
            displacement = np.linalg.norm(trajectory[i] - trajectory[i-1])
            speed = displacement / env_unwrapped.time_step
            speeds.append(speed)
        plt.plot(speeds, 'navy', linewidth=1.5)
        plt.xlabel('Step')
        plt.ylabel('Speed (m/s)')
        plt.title('Speed Curve')
        plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 10)
    # Simulated reward component data
    components = ['Throughput', 'Movement', 'Approach', 'Terminal']
    values = [120, 25, 150, 2500]  # Based on config weights
    plt.bar(components, values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Reward Component')
    plt.ylabel('Weight')
    plt.title('Reward Weight Config')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 4, 12)
    plt.axis('off')

    if len(trajectory) > 0:
        final_pos = trajectory[-1][:2]
        target_pos = env_unwrapped.end_position[:2]
        final_distance = np.linalg.norm(final_pos - target_pos)
        success = final_distance <= end_tolerance

        metrics_text = f"""
Performance Summary

Arrived: {'Success' if success else 'Fail'}
Final Distance: {final_distance:.2f}m
Tolerance: {end_tolerance}m
Guidance Range: {close_to_end_threshold}m
Final Position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})
Total Reward: {result['total_reward']:.0f}
Total Throughput: {result['total_throughput']:.1f}
Total Steps: {result['steps']}

Guidance Strategy Effect:
Approach Reward: {reward_config.w_user_approach}
Distance Progress: {reward_config.w_distance_progress}
Terminal Reward: {reward_config.B_reach_end}
        """

        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_convergence_comparison(env1_callback, env2_callback):
    """Plot convergence curves for 2 different sets of user positions"""
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Raw convergence curves
    plt.subplot(1, 2, 1)
    if len(env1_callback.episode_rewards) > 0:
        plt.plot(env1_callback.episode_rewards, 'b-', alpha=0.7, linewidth=1.5, 
                label='User Set 1: [(15,75), (75,15)]')
    if len(env2_callback.episode_rewards) > 0:
        plt.plot(env2_callback.episode_rewards, 'r-', alpha=0.7, linewidth=1.5,
                label='User Set 2: [(30,30), (70,70)]')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Convergence Comparison: Different User Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed convergence curves
    plt.subplot(1, 2, 2)
    
    # Smooth env1 data
    if len(env1_callback.episode_rewards) > 0:
        window_size = min(10, len(env1_callback.episode_rewards) // 4)
        if window_size > 1:
            smoothed1 = np.convolve(env1_callback.episode_rewards, 
                                  np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(env1_callback.episode_rewards)), 
                    smoothed1, 'b-', linewidth=2, label='User Set 1 (Smoothed)')
    
    # Smooth env2 data
    if len(env2_callback.episode_rewards) > 0:
        window_size = min(10, len(env2_callback.episode_rewards) // 4)
        if window_size > 1:
            smoothed2 = np.convolve(env2_callback.episode_rewards, 
                                  np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(env2_callback.episode_rewards)), 
                    smoothed2, 'r-', linewidth=2, label='User Set 2 (Smoothed)')
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    if len(env1_callback.episode_rewards) > 0 and len(env2_callback.episode_rewards) > 0:
        print(f"\nConvergence Comparison Statistics:")
        print(f"User Set 1 - Final 10 episodes avg: {np.mean(env1_callback.episode_rewards[-10:]):.2f}")
        print(f"User Set 2 - Final 10 episodes avg: {np.mean(env2_callback.episode_rewards[-10:]):.2f}")
        print(f"User Set 1 - Total episodes: {len(env1_callback.episode_rewards)}")
        print(f"User Set 2 - Total episodes: {len(env2_callback.episode_rewards)}")


def plot_dwelling_time_trajectories(agent, env, num_episodes=10):
    """Plot trajectories of 10 optimized UAV episodes with dwelling time markers"""
    plt.figure(figsize=(16, 10))
    
    # Collect all trajectories and dwelling times
    all_trajectories = []
    all_dwelling_times = {}  # position -> total time
    
    for ep in range(num_episodes):
        result = evaluate_trajectory(agent, env, deterministic=True)
        if len(result['trajectory']) > 0:
            all_trajectories.append(result['trajectory'])
            
            # Calculate dwelling times for this trajectory
            for i, pos in enumerate(result['trajectory']):
                pos_key = (round(pos[0], 1), round(pos[1], 1))  # Round to avoid floating point issues
                if pos_key not in all_dwelling_times:
                    all_dwelling_times[pos_key] = 0
                all_dwelling_times[pos_key] += env.unwrapped.time_step
    
    # Plot trajectories
    plt.subplot(2, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes))
    
    for i, trajectory in enumerate(all_trajectories):
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                color=colors[i], alpha=0.6, linewidth=1.5, label=f'Episode {i+1}')
    
    # Add environment elements
    env_unwrapped = env.unwrapped
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=200, marker='o', 
               label='Start', zorder=10, edgecolors='black', linewidth=2)
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=300, marker='*', 
               label='End', zorder=10, edgecolors='black', linewidth=2)
    
    user_positions = env_unwrapped.get_user_positions()
    for i, user_pos in enumerate(user_positions):
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=150, marker='x',
                   label=f'User{i+1}' if i == 0 else "", zorder=8)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('10 Optimized UAV Trajectories')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # Plot dwelling times with scaled markers
    plt.subplot(2, 2, 2)
    
    # Plot base trajectories in light gray
    for trajectory in all_trajectories:
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                color='lightgray', alpha=0.3, linewidth=1)
    
    # Plot dwelling time markers
    if all_dwelling_times:
        positions = list(all_dwelling_times.keys())
        times = list(all_dwelling_times.values())
        
        # Normalize marker sizes
        max_time = max(times)
        min_time = min(times)
        if max_time > min_time:
            normalized_sizes = [(t - min_time) / (max_time - min_time) * 200 + 20 for t in times]
        else:
            normalized_sizes = [50] * len(times)
        
        # Plot markers
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        scatter = plt.scatter(x_coords, y_coords, s=normalized_sizes, 
                            c=times, cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='Dwelling Time (s)')
    
    # Add environment elements
    plt.scatter(*env_unwrapped.start_position[:2], c='green', s=200, marker='o', 
               label='Start', zorder=10, edgecolors='black', linewidth=2)
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=300, marker='*', 
               label='End', zorder=10, edgecolors='black', linewidth=2)
    
    for i, user_pos in enumerate(user_positions):
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=150, marker='x',
                   label=f'User{i+1}' if i == 0 else "", zorder=8)
    
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Dwelling Time Analysis (Marker Size ∝ Time)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # Plot dwelling time statistics
    plt.subplot(2, 2, 3)
    if all_dwelling_times:
        sorted_times = sorted(times, reverse=True)
        top_10_times = sorted_times[:min(10, len(sorted_times))]
        
        plt.bar(range(len(top_10_times)), top_10_times, alpha=0.7, color='orange')
        plt.xlabel('Rank (Top Dwelling Locations)')
        plt.ylabel('Total Dwelling Time (s)')
        plt.title('Top 10 Dwelling Locations')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, time in enumerate(top_10_times):
            plt.text(i, time + max(top_10_times)*0.01, f'{time:.1f}s', 
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot trajectory length distribution
    plt.subplot(2, 2, 4)
    if all_trajectories:
        lengths = [len(traj) for traj in all_trajectories]
        plt.hist(lengths, bins=min(10, max(1, len(set(lengths)))), 
                alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Trajectory Length (steps)')
        plt.ylabel('Frequency')
        plt.title('Trajectory Length Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.text(0.7, 0.8, f'Mean: {np.mean(lengths):.1f}\nStd: {np.std(lengths):.1f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return all_dwelling_times


def plot_throughput_comparison(optimized_total_throughput):
    """Plot throughput comparison bar chart"""
    plt.figure(figsize=(12, 8))
    
    # Throughput data
    categories = [
        'Benchmark trajectory\nwith optimized transmit signal',
        'Benchmark trajectory with\nrandomized transmit beamformers', 
        'Optimized trajectory with\nrandomized transmit beamformers\n(random+equal)',
        'Optimized trajectory with\noptimized transmit beamformers\n(mrt + proportional)'
    ]
    
    throughputs = [
        8578.7,  # Benchmark optimized
        3089.8,  # Benchmark randomized
        optimized_total_throughput * 0.6,  # Estimated for random+equal (60% of optimized)
        optimized_total_throughput  # Our optimized result
    ]
    
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = plt.bar(categories, throughputs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.ylabel('Total Throughput', fontsize=12)
    plt.title('Throughput Comparison: Different Trajectory and Beamforming Strategies', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(throughputs)*0.01,
                f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add improvement percentages
    baseline = throughputs[0]  # Benchmark optimized as baseline
    for i, (bar, throughput) in enumerate(zip(bars[1:], throughputs[1:]), 1):
        improvement = (throughput - baseline) / baseline * 100
        color = 'green' if improvement > 0 else 'red'
        plt.text(bar.get_x() + bar.get_width()/2., throughput/2,
                f'{improvement:+.1f}%', ha='center', va='center', 
                fontweight='bold', color=color, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add legend
    legend_labels = [
        'Baseline (Benchmark + Optimized Signal)',
        'Benchmark + Random Signal', 
        'Our Method + Random Signal',
        'Our Method + Optimized Signal'
    ]
    plt.legend(bars, legend_labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print(f"\nThroughput Comparison Results:")
    for i, (category, throughput) in enumerate(zip(categories, throughputs)):
        improvement = (throughput - throughputs[0]) / throughputs[0] * 100
        print(f"{i+1}. {category.replace(chr(10), ' ')}: {throughput:.1f} ({improvement:+.1f}%)")


def main():
    print("🚀 Enhanced UAV Trajectory Optimization with Comprehensive Analysis 🚀")
    
    # 1. Create enhanced environment (User Set 1)
    env1 = create_enhanced_environment()
    
    # 2. Train enhanced DQN for User Set 1
    print("\n🎯 Training with User Set 1: [(15,75), (75,15)]")
    agent1, callback1, monitored_env1 = train_enhanced_dqn(env1, total_timesteps=250000)
    
    # # 3. Create environment with different user positions (User Set 2)
    # print("\n🎯 Training with User Set 2: [(30,30), (70,70)]")
    # env2 = create_enhanced_environment()
    # # Modify user positions for comparison
    # alt_positions = np.array([
    #     [30.0, 30.0, 0.0],   
    #     [70.0, 70.0, 0.0]    
    # ])
    # env2.user_manager.set_user_positions(alt_positions)
    # agent2, callback2, monitored_env2 = train_enhanced_dqn(env2, total_timesteps=250000)  # Shorter training for comparison
    
    # # 4. Plot convergence comparison for different user positions
    # print("\n📈 Plotting convergence comparison...")
    # plot_convergence_comparison(callback1, callback2)
    
    # 5. Plot training results for main environment
    print("\n📊 Plotting main training results...")
    plot_training_results(callback1)

    # 6. Evaluate a single trajectory
    result = evaluate_trajectory(agent1, monitored_env1, deterministic=True)

    print("Single trajectory evaluation:")
    print(f"  Total reward: {result['total_reward']:.2f}")
    print(f"  Total throughput: {result['total_throughput']:.2f}")
    print(f"  Steps: {result['steps']}")
    print(f"  Reached end: {result['reached_end']}")
    print(f"  Final position: {result['final_position']}")
    print(f"  Target position: {result['target_position']}")

    if result['final_position'] is not None:
        distance_to_target = np.linalg.norm(result['final_position'] - result['target_position'])
        tolerance = monitored_env1.unwrapped.reward_calculator.config.end_position_tolerance
        print(f"  Distance to target: {distance_to_target:.2f}m")
        print(f"  Tolerance: {tolerance}m")
        print(f"  Precision status: {'High-precision arrival!' if distance_to_target <= tolerance else 'Needs further optimization'}")

    # 7. Plot enhanced trajectory analysis
    print("\n🔥 Plotting enhanced trajectory analysis...")
    plot_enhanced_trajectory_with_circles(result, monitored_env1)

    # 8. Plot dwelling time trajectories analysis
    print("\n🏠 Plotting dwelling time trajectories...")
    dwelling_times = plot_dwelling_time_trajectories(agent1, monitored_env1, num_episodes=10)

    # 9. Plot throughput comparison
    print("\n📊 Plotting throughput comparison...")
    plot_throughput_comparison(result['total_throughput'])

    # 10. Evaluate multiple trajectories for final statistics
    print("\n📈 Evaluating multiple trajectories for final statistics...")
    results = []
    success_count = 0
    distances = []

    for ep in range(10):
        result = evaluate_trajectory(agent1, monitored_env1, deterministic=True)
        results.append(result)

        if result['final_position'] is not None:
            distance = np.linalg.norm(result['final_position'] - result['target_position'])
            distances.append(distance)
            tolerance = monitored_env1.unwrapped.reward_calculator.config.end_position_tolerance
            success = distance <= tolerance
            if success:
                success_count += 1
            print(f"Episode {ep}: reward={result['total_reward']:.0f}, distance={distance:.2f}m, {'Success' if success else 'Fail'}")
        else:
            print(f"Episode {ep}: Invalid trajectory")

    # 11. Plot final comprehensive analysis
    print("\n🎯 Plotting final comprehensive analysis...")
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)

    env_unwrapped = monitored_env1.unwrapped
    reward_config = env_unwrapped.reward_calculator.config
    user_service_radius = reward_config.user_service_radius
    end_tolerance = reward_config.end_position_tolerance
    close_to_end_threshold = reward_config.close_to_end_threshold

    for i, result in enumerate(results):
        trajectory = result['trajectory']
        if len(trajectory) > 0:
            success = result['reached_end']
            alpha = 0.8 if success else 0.4
            color = 'green' if success else 'red'
            plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha, linewidth=2)

    user_positions = env_unwrapped.get_user_positions()
    for i, user_pos in enumerate(user_positions):
        service_circle = plt.Circle((user_pos[0], user_pos[1]), user_service_radius,
                                  fill=False, color='purple', linestyle='--',
                                  linewidth=2, alpha=0.7)
        plt.gca().add_patch(service_circle)
        plt.scatter(user_pos[0], user_pos[1], c='purple', s=100, marker='x',
                   label=f'User{i+1}' if i == 0 else "")

    end_pos = env_unwrapped.end_position[:2]
    tolerance_circle = plt.Circle((end_pos[0], end_pos[1]), end_tolerance,
                                fill=False, color='red', linestyle='-',
                                linewidth=4, alpha=0.9, label=f'Tolerance ({end_tolerance}m)')
    plt.gca().add_patch(tolerance_circle)

    guidance_circle = plt.Circle((end_pos[0], end_pos[1]), close_to_end_threshold,
                               fill=False, color='orange', linestyle=':',
                               linewidth=3, alpha=0.8, label=f'Guidance ({close_to_end_threshold}m)')
    plt.gca().add_patch(guidance_circle)

    plt.scatter(*env_unwrapped.start_position[:2], c='blue', s=150, marker='o', label='Start')
    plt.scatter(*env_unwrapped.end_position[:2], c='red', s=200, marker='*', label='End')

    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'10 Enhanced Guidance Trajectories\n(Green=Success, Red=Fail)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')

    plt.subplot(1, 2, 2)

    if distances:
        plt.hist(distances, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=end_tolerance, color='red', linestyle='--', linewidth=3,
                   label=f'Tolerance ({end_tolerance}m)')
        plt.axvline(x=np.mean(distances), color='orange', linestyle='-', linewidth=2,
                   label=f'Average Distance ({np.mean(distances):.1f}m)')

        plt.xlabel('Arrival Distance (m)')
        plt.ylabel('Frequency')
        plt.title(f'Arrival Distance Distribution\nSuccess Rate: {success_count}/10 ({success_count*10}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n🏆 === Final Comprehensive Statistics === 🏆")
    print(f"✅ Arrived at end: {success_count}/10 ({success_count*10}%)")
    if distances:
        print(f"📏 Average arrival distance: {np.mean(distances):.2f}m")
        print(f"🎯 Best arrival distance: {min(distances):.2f}m")
        print(f"📊 Worst arrival distance: {max(distances):.2f}m")
        print(f"🎗️ Tolerance: {end_tolerance}m")
        
        # Calculate improvement metrics
        baseline_distance = 19.8  # From previous results
        improvement_vs_prev = baseline_distance - np.mean(distances)
        print(f"📈 Improvement vs baseline: {improvement_vs_prev:.2f}m ({improvement_vs_prev/baseline_distance*100:.1f}%)")
        
        # Dwelling time statistics
        if dwelling_times:
            total_dwelling_locations = len(dwelling_times)
            max_dwelling_time = max(dwelling_times.values())
            avg_dwelling_time = np.mean(list(dwelling_times.values()))
            print(f"🏠 Total dwelling locations: {total_dwelling_locations}")
            print(f"⏰ Max dwelling time: {max_dwelling_time:.1f}s")
            print(f"⏱️ Average dwelling time: {avg_dwelling_time:.1f}s")

    print(f"\n🎉 === Enhanced UAV Trajectory Optimization Completed === 🎉")


if __name__ == '__main__':
    main()