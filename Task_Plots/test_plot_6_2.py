"""
信号功率与发射功率预算关系图绘制工具

此模块提供绘制接收信号功率随发射功率预算变化关系图的功能。
支持不同用户数量K = 1,2,3,4的对比分析。

环境限制：
- UAV固定在z=50m高度
- User固定在z=0m地面
- 考虑噪声影响的接收功率计算

主要函数：
- plot_signal_power_vs_transmit_power(): 计算单个K值的结果
- plot_6_2(): 绘制所有K值的综合对比图
- main(): 主函数，完整流程

使用示例：
    # 直接运行主函数
    python plot_signal_power_vs_transmit_power.py
    
    # 或者导入使用
    from plot_signal_power_vs_transmit_power import main
    all_results, figure = main()
"""
import sys
import os
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.utils.channel import ChannelModel
from src.environment.uav import UAV
from src.environment.users import GroundUser


def plot_signal_power_vs_transmit_power(K=1,
                                      env_bounds=(100, 100, 50),
                                      user_positions=None,
                                      uav_position=(0, 0, 50),
                                      transmit_power_range=(0.1, 2.0),
                                      num_power_points=50,
                                      frequency=2.4e9,
                                      path_loss_exponent=2.5,
                                      save_path=None):
    """
    计算接收信号功率与发射功率预算的关系
    
    Args:
        K: 用户数量 (1, 2, 3, 4)
        env_bounds: 环境边界 (x_max, y_max, z_max)
        user_positions: 用户位置列表，如果为None则自动生成
        uav_position: UAV位置 (x, y, z)
        transmit_power_range: 发射功率范围 (min, max) in Watts
        num_power_points: 功率采样点数
        frequency: 载波频率 (Hz)
        path_loss_exponent: 路径损耗指数 η
        save_path: 保存路径，如果为None则不保存图片
    
    Returns:
        dict: 包含功率、接收功率数据的字典
    """
    
    # 初始化UAV
    uav = UAV(
        start_position=uav_position,
        transmit_power=transmit_power_range[1],  # 使用最大功率初始化
        env_bounds=env_bounds
    )
    
    # 生成或使用用户位置
    if user_positions is None:
        # 自动生成K个用户位置，均匀分布在环境内
        user_positions = []
        if K == 1:
            user_positions = [(50, 50, 0)]  # 环境中心
        elif K == 2:
            user_positions = [(25, 25, 0), (75, 75, 0)]  # 对角线分布
        elif K == 3:
            user_positions = [(25, 25, 0), (75, 25, 0), (50, 75, 0)]  # 三角形分布
        elif K == 4:
            user_positions = [(25, 25, 0), (75, 25, 0), (25, 75, 0), (75, 75, 0)]  # 四角分布
    
    print(f"创建了{K}个用户")
    print(f"用户位置: {user_positions}")
    
    # 创建发射功率范围
    transmit_powers = np.linspace(transmit_power_range[0], 
                                 transmit_power_range[1], 
                                 num_power_points)
    
    # 初始化信道模型
    channel_model = ChannelModel(
        frequency=frequency,
        path_loss_exponent=path_loss_exponent,
        noise_power=-100.0  # dB
    )
    
    # 计算每个用户的信道系数（固定，不随发射功率变化）
    channel_coeffs = []
    distances = []
    
    for pos in user_positions:
        user_pos = np.array(pos)
        distance = uav.distance_to(user_pos)
        distances.append(distance)
        
        # 计算信道系数
        channel_coeff = channel_model.calculate_channel_coefficient(
            uav.get_position(), 
            user_pos
        )
        channel_coeffs.append(channel_coeff)
    
    channel_coeffs = np.array(channel_coeffs)
    distances = np.array(distances)
    
    # 计算不同发射功率下的接收功率
    received_powers_total = []  # 总接收功率（所有用户）
    received_powers_individual = []  # 每个用户的接收功率
    snr_values = []
    
    for tx_power in transmit_powers:
        # 计算每个用户的接收功率
        user_powers = []
        user_snrs = []
        
        for i, channel_coeff in enumerate(channel_coeffs):
            # 信号功率
            signal_power = tx_power * (np.abs(channel_coeff) ** 2)
            # 接收功率 = 信号功率 + 噪声功率
            received_power = signal_power #+ channel_model.noise_power
            user_powers.append(received_power)
            
            # SNR
            snr = channel_model.calculate_snr(channel_coeff, tx_power)
            user_snrs.append(snr)
        
        # 总接收功率（所有用户的和）
        total_power = np.sum(user_powers)
        received_powers_total.append(total_power)
        received_powers_individual.append(user_powers)
        snr_values.append(user_snrs)
    
    received_powers_total = np.array(received_powers_total)
    received_powers_individual = np.array(received_powers_individual)
    snr_values = np.array(snr_values)
    
    # 转换为dB
    received_powers_total_db = 10 * np.log10(received_powers_total + 1e-20)
    received_powers_individual_db = 10 * np.log10(received_powers_individual + 1e-20)
    snr_values_db = 10 * np.log10(snr_values + 1e-20)
    transmit_powers_db = 10 * np.log10(transmit_powers + 1e-20)
    
    # 存储结果
    results = {
        'transmit_powers': transmit_powers,
        'transmit_powers_db': transmit_powers_db,
        'received_powers_total_db': received_powers_total_db,
        'received_powers_individual_db': received_powers_individual_db,
        'snr_values_db': snr_values_db,
        'user_positions': user_positions,
        'distances': distances,
        'channel_coeffs': channel_coeffs,
        'K': K
    }
    
    return results


def plot_6_2(all_results, save_path="results/signal_power_vs_transmit_power_all_K.png"):
    """
    绘制所有K值的信号功率与发射功率关系图在一张图上
    
    Args:
        all_results: 字典，键为K值，值为对应的结果字典
        save_path: 保存路径
    """
    
    # 设置绘图
    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    K_values = sorted(all_results.keys())  # 按K值排序
    
    for i, K in enumerate(K_values):
        results = all_results[K]
        # transmit_powers_db = results['transmit_powers_db']
        transmit_powers = results['transmit_powers']
        received_powers_db = results['received_powers_total_db']
        
        # 绘制曲线
        plt.plot(transmit_powers, received_powers_db, 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                markersize=4, linewidth=2.5,
                label=f'K = {K} users')
    
    # 图形美化
    # plt.xlabel('Transmit Power Budget (dB)', fontsize=13, fontweight='bold')
    plt.xlabel('Transmit Power Budget (W)', fontsize=13, fontweight='bold')
    plt.ylabel('Total Received Signal Power (dB)', fontsize=13, fontweight='bold')
    plt.title('Signal Power vs. Transmit Power Budget\n'
              'UAV at z=50m, Users at z=0m with Different User Numbers', 
              fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    
    # 添加对数坐标轴（可选，因为功率关系通常是对数的）
    plt.xscale('log')
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"发射功率关系图已保存到: {save_path}")
    else:
        plt.show()
    
    return plt.gcf()  # 返回图形对象


def main():
    """
    主函数：为不同的K值调用plot_signal_power_vs_transmit_power函数，
    然后调用plot_6_2函数绘制综合图表
    """
    
    # 参数设置
    K_values = [1, 2, 3, 4]
    env_bounds = (100, 100, 50)
    uav_position = (0, 0, 50)  # UAV在原点上方50m
    transmit_power_range = (0.1, 2.0)  # 0.1W 到 2.0W
    num_power_points = 50
    frequency = 2.4e9
    path_loss_exponent = 2.5
    
    print(f"开始计算不同K值的信号功率与发射功率关系...")
    print(f"K值列表: {K_values}")
    print(f"环境边界: {env_bounds}")
    print(f"UAV位置: {uav_position}")
    print(f"发射功率范围: {transmit_power_range[0]} - {transmit_power_range[1]} W")
    print(f"路径损耗指数: η = {path_loss_exponent}")
    print()
    
    # 收集所有K值的结果
    all_results = {}
    
    for K in K_values:
        print(f"正在计算 K = {K}...")
        
        # 调用函数
        results = plot_signal_power_vs_transmit_power(
            K=K,
            env_bounds=env_bounds,
            uav_position=uav_position,
            transmit_power_range=transmit_power_range,
            num_power_points=num_power_points,
            frequency=frequency,
            path_loss_exponent=path_loss_exponent,
            save_path=None  # 不保存单独的图，只保存最终综合图
        )
        
        # 存储结果
        all_results[K] = results
        
        # 显示一些统计信息
        tx_powers = results['transmit_powers']
        rx_powers_db = results['received_powers_total_db']
        print(f"  发射功率范围: {tx_powers.min():.2f} - {tx_powers.max():.2f} W")
        print(f"  接收功率范围: {rx_powers_db.min():.1f} - {rx_powers_db.max():.1f} dB")
        print(f"  用户数量: {results['K']}")
        print(f"  用户位置: {results['user_positions']}")
        print()
    
    print("所有K值计算完成，开始绘制综合图表...")
    
    # 调用plot_6_2函数绘制综合图表
    fig = plot_6_2(
        all_results=all_results,
        save_path="results/signal_power_vs_transmit_power_all_K.png"
    )
    
    print("绘图完成！")
    
    # 返回结果供进一步分析
    return all_results, fig


if __name__ == "__main__":
    # 运行主函数
    all_results, figure = main()
    
    # 可选：显示一些额外的统计信息
    print("\n=== 总结统计 ===")
    for K in sorted(all_results.keys()):
        results = all_results[K]
        tx_powers = results['transmit_powers']
        rx_powers_db = results['received_powers_total_db']
        
        # 计算功率效率（接收功率/发射功率）
        power_efficiency = results['received_powers_total_db'] - 10 * np.log10(tx_powers)
        
        print(f"K = {K}:")
        print(f"  平均接收功率: {rx_powers_db.mean():.1f} dB")
        print(f"  功率效率: {power_efficiency.mean():.1f} dB/W")
        print(f"  用户位置: {results['user_positions']}")
