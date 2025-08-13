#!/usr/bin/env python3
"""
完整的蒙特卡洛仿真：信道平均 + Random方法平均
包含丰富的可视化分析
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 导入模块
from environment.uav_env import UAVEnvironment

def test_channel_signal_integration_complete(num_channel_trials=20, num_random_trials=50, base_seed=0):
    """完整的蒙特卡洛仿真：信道平均 + Random方法平均"""
    
    print("🔧 完整蒙特卡洛仿真：信道平均 + Random方法平均...")
    
    beamforming_methods = ['mrt', 'zf', 'random']
    power_strategies = ['equal', 'proportional', 'water_filling']
    
    results = {}
    detailed_results = {}  # 存储详细数据用于可视化
    
    for bf_method in beamforming_methods:
        results[bf_method] = {}
        detailed_results[bf_method] = {}
        
        for power_strategy in power_strategies:
            channel_means = []
            channel_data = []  # 存储每个信道的详细数据
            
            print(f"\n--- 测试 {bf_method.upper()} + {power_strategy} ---")
            
            # 外层：不同信道实现
            for c in range(num_channel_trials):
                env = UAVEnvironment(seed=base_seed + c)
                env.reset(seed=base_seed + c)
                env.transmit_power = 10
                
                if bf_method == 'random':
                    # Random方法：在同一信道下多次平均
                    random_vals = []
                    for r in range(num_random_trials):
                        thr = env._calculate_throughput(bf_method, power_strategy)
                        random_vals.append(thr)
                    
                    channel_mean = np.mean(random_vals)
                    channel_std = np.std(random_vals)
                    channel_means.append(channel_mean)
                    channel_data.append({
                        'mean': channel_mean,
                        'std': channel_std,
                        'values': random_vals
                    })
                    
                    if c < 3:  # 显示前3个信道的结果
                        print(f"   信道{c+1}: {channel_mean:.4f} ± {channel_std:.4f}")
                else:
                    # MRT/ZF：确定性方法，单次计算即可
                    thr = env._calculate_throughput(bf_method, power_strategy)
                    channel_means.append(thr)
                    channel_data.append({
                        'mean': thr,
                        'std': 0.0,
                        'values': [thr]
                    })
                    
                    if c < 3:  # 显示前3个信道的结果
                        print(f"   信道{c+1}: {thr:.4f}")
            
            # 最终结果：跨信道平均
            final_mean = float(np.mean(channel_means))
            final_std = float(np.std(channel_means))
            
            if bf_method == 'random':
                print(f"   最终: {final_mean:.4f} ± {final_std:.4f} (n={num_channel_trials} channels, {num_random_trials} random trials each)")
            else:
                print(f"   最终: {final_mean:.4f} ± {final_std:.4f} (n={num_channel_trials} channels)")
            
            results[bf_method][power_strategy] = final_mean
            detailed_results[bf_method][power_strategy] = {
                'mean': final_mean,
                'std': final_std,
                'channel_means': channel_means,
                'channel_data': channel_data
            }
    
    return results, detailed_results

def create_comprehensive_visualizations(results: Dict, detailed_results: Dict, num_channel_trials: int, num_random_trials: int):
    """创建综合可视化"""
    
    beamforming_methods = ['mrt', 'zf', 'random']
    power_strategies = ['equal', 'proportional', 'water_filling']
    
    # 设置绘图样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建2x3的子图布局
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 平均性能对比柱状图
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(beamforming_methods))
    width = 0.25
    
    for i, strategy in enumerate(power_strategies):
        means = [detailed_results[method][strategy]['mean'] for method in beamforming_methods]
        stds = [detailed_results[method][strategy]['std'] for method in beamforming_methods]
        
        bars = ax1.bar(x + i*width, means, width, label=strategy, alpha=0.8)
        
        # 添加误差条
        ax1.errorbar(x + i*width, means, yerr=stds, fmt='none', color='black', capsize=5)
        
        # 在柱子上添加数值标签
        for j, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(x[j] + i*width, mean + std + 0.1, f'{mean:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Beamforming Method')
    ax1.set_ylabel('System Throughput')
    ax1.set_title('Average Performance Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([m.upper() for m in beamforming_methods])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 标准差对比
    ax2 = plt.subplot(2, 3, 2)
    for i, strategy in enumerate(power_strategies):
        stds = [detailed_results[method][strategy]['std'] for method in beamforming_methods]
        ax2.bar(x + i*width, stds, width, label=strategy, alpha=0.8)
    
    ax2.set_xlabel('Beamforming Method')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Performance Variability')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([m.upper() for m in beamforming_methods])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信道间性能分布（箱线图）
    ax3 = plt.subplot(2, 3, 3)
    data_for_boxplot = []
    labels_for_boxplot = []
    
    for method in beamforming_methods:
        for strategy in power_strategies:
            channel_means = detailed_results[method][strategy]['channel_means']
            data_for_boxplot.append(channel_means)
            labels_for_boxplot.append(f'{method.upper()}\n{strategy}')
    
    box_plot = ax3.boxplot(data_for_boxplot, tick_labels=labels_for_boxplot, patch_artist=True)
    
    # 设置不同颜色
    colors = ['lightblue', 'lightgreen', 'lightcoral'] * len(beamforming_methods)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Throughput')
    ax3.set_title('Performance Distribution Across Channels')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Random方法的收敛性分析
    ax4 = plt.subplot(2, 3, 4)
    for strategy in power_strategies:
        channel_data = detailed_results['random'][strategy]['channel_data']
        
        # 计算每个信道的收敛性
        convergence_data = []
        for channel_info in channel_data:
            values = channel_info['values']
            cumulative_means = [np.mean(values[:i+1]) for i in range(len(values))]
            convergence_data.append(cumulative_means)
        
        # 计算平均收敛曲线
        avg_convergence = np.mean(convergence_data, axis=0)
        std_convergence = np.std(convergence_data, axis=0)
        
        trials = np.arange(1, len(avg_convergence) + 1)
        ax4.plot(trials, avg_convergence, label=f'Random + {strategy}', linewidth=2)
        ax4.fill_between(trials, avg_convergence - std_convergence, 
                        avg_convergence + std_convergence, alpha=0.3)
    
    ax4.set_xlabel('Number of Random Trials')
    ax4.set_ylabel('Cumulative Average Throughput')
    ax4.set_title('Random Beamforming Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 性能排序热力图
    ax5 = plt.subplot(2, 3, 5)
    
    # 创建性能矩阵
    performance_matrix = np.zeros((len(beamforming_methods), len(power_strategies)))
    for i, method in enumerate(beamforming_methods):
        for j, strategy in enumerate(power_strategies):
            performance_matrix[i, j] = detailed_results[method][strategy]['mean']
    
    # 创建热力图
    im = ax5.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
    
    # 添加数值标签
    for i in range(len(beamforming_methods)):
        for j in range(len(power_strategies)):
            text = ax5.text(j, i, f'{performance_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax5.set_xticks(range(len(power_strategies)))
    ax5.set_yticks(range(len(beamforming_methods)))
    ax5.set_xticklabels([s.upper() for s in power_strategies])
    ax5.set_yticklabels([m.upper() for m in beamforming_methods])
    ax5.set_title('Performance Heatmap')
    
    # 添加颜色条
    plt.colorbar(im, ax=ax5, label='Throughput')
    
    # 6. 统计显著性分析
    ax6 = plt.subplot(2, 3, 6)
    
    # 计算性能差异的统计显著性
    method_comparisons = []
    for i, method1 in enumerate(beamforming_methods):
        for j, method2 in enumerate(beamforming_methods):
            if i < j:  # 避免重复比较
                for strategy in power_strategies:
                    data1 = detailed_results[method1][strategy]['channel_means']
                    data2 = detailed_results[method2][strategy]['channel_means']
                    
                    # 简单的t检验（这里用均值差异作为显著性指标）
                    mean_diff = np.mean(data1) - np.mean(data2)
                    pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                    t_stat = mean_diff / (pooled_std * np.sqrt(2/len(data1)))
                    
                    method_comparisons.append({
                        'comparison': f'{method1.upper()} vs {method2.upper()}',
                        'strategy': strategy,
                        't_stat': abs(t_stat),
                        'mean_diff': mean_diff
                    })
    
    # 绘制t统计量
    comparisons = [f"{comp['comparison']}\n{comp['strategy']}" for comp in method_comparisons]
    t_stats = [comp['t_stat'] for comp in method_comparisons]
    
    bars = ax6.bar(range(len(comparisons)), t_stats, alpha=0.7)
    
    # 添加显著性阈值线
    ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Significance threshold (|t|>2)')
    
    ax6.set_xlabel('Method Comparisons')
    ax6.set_ylabel('|t-statistic|')
    ax6.set_title('Statistical Significance Analysis')
    ax6.set_xticks(range(len(comparisons)))
    ax6.set_xticklabels(comparisons, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 创建额外的详细分析图
    create_detailed_analysis_plots(detailed_results, num_channel_trials, num_random_trials)

def create_detailed_analysis_plots(detailed_results: Dict, num_channel_trials: int, num_random_trials: int):
    """创建详细分析图"""
    
    beamforming_methods = ['mrt', 'zf', 'random']
    power_strategies = ['equal', 'proportional', 'water_filling']
    
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 各方法在不同信道上的性能表现
    ax1 = axes[0, 0]
    for method in beamforming_methods:
        for strategy in power_strategies:
            channel_means = detailed_results[method][strategy]['channel_means']
            channels = range(1, len(channel_means) + 1)
            ax1.plot(channels, channel_means, 'o-', 
                    label=f'{method.upper()} + {strategy}', alpha=0.7, markersize=4)
    
    ax1.set_xlabel('Channel Realization')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Performance Across Different Channel Realizations')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 性能提升百分比
    ax2 = axes[0, 1]
    
    # 以Random + equal为基准
    baseline = detailed_results['random']['equal']['mean']
    
    improvements = {}
    for method in beamforming_methods:
        for strategy in power_strategies:
            current = detailed_results[method][strategy]['mean']
            improvement = ((current - baseline) / baseline) * 100
            improvements[f'{method.upper()}_{strategy}'] = improvement
    
    methods = list(improvements.keys())
    values = list(improvements.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax2.bar(methods, values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    ax2.set_xlabel('Method + Strategy')
    ax2.set_ylabel('Performance Improvement (%)')
    ax2.set_title('Performance Improvement Relative to Random + Equal')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能稳定性分析（变异系数）
    ax3 = axes[1, 0]
    
    cv_data = {}
    for method in beamforming_methods:
        for strategy in power_strategies:
            mean_val = detailed_results[method][strategy]['mean']
            std_val = detailed_results[method][strategy]['std']
            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
            cv_data[f'{method.upper()}_{strategy}'] = cv
    
    methods_cv = list(cv_data.keys())
    cv_values = list(cv_data.values())
    
    bars = ax3.bar(methods_cv, cv_values, alpha=0.7, color='orange')
    
    # 添加数值标签
    for bar, value in zip(bars, cv_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom')
    
    ax3.set_xlabel('Method + Strategy')
    ax3.set_ylabel('Coefficient of Variation (%)')
    ax3.set_title('Performance Stability (Lower is Better)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合性能评分
    ax4 = axes[1, 1]
    
    # 计算综合评分（考虑性能和稳定性）
    scores = {}
    for method in beamforming_methods:
        for strategy in power_strategies:
            mean_val = detailed_results[method][strategy]['mean']
            std_val = detailed_results[method][strategy]['std']
            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
            
            # 综合评分 = 性能得分 - 稳定性惩罚
            performance_score = mean_val / max([detailed_results[m][s]['mean'] 
                                              for m in beamforming_methods 
                                              for s in power_strategies]) * 100
            stability_penalty = cv * 0.5  # 稳定性惩罚权重
            total_score = performance_score - stability_penalty
            
            scores[f'{method.upper()}_{strategy}'] = total_score
    
    methods_score = list(scores.keys())
    score_values = list(scores.values())
    
    bars = ax4.bar(methods_score, score_values, alpha=0.7, color='purple')
    
    # 添加数值标签
    for bar, value in zip(bars, score_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    ax4.set_xlabel('Method + Strategy')
    ax4.set_ylabel('Comprehensive Score')
    ax4.set_title('Comprehensive Performance Score\n(Performance - Stability Penalty)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_statistical_summary(results: Dict, detailed_results: Dict):
    """打印统计摘要"""
    
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    
    beamforming_methods = ['mrt', 'zf', 'random']
    power_strategies = ['equal', 'proportional', 'water_filling']
    
    # 创建结果表格
    print(f"{'Method':<15} {'Strategy':<15} {'Mean':<10} {'Std':<10} {'CV(%)':<10} {'Rank':<5}")
    print("-" * 80)
    
    all_results = []
    for method in beamforming_methods:
        for strategy in power_strategies:
            mean_val = detailed_results[method][strategy]['mean']
            std_val = detailed_results[method][strategy]['std']
            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
            
            all_results.append({
                'method': method.upper(),
                'strategy': strategy,
                'mean': mean_val,
                'std': std_val,
                'cv': cv
            })
    
    # 按均值排序
    all_results.sort(key=lambda x: x['mean'], reverse=True)
    
    for i, result in enumerate(all_results):
        print(f"{result['method']:<15} {result['strategy']:<15} "
              f"{result['mean']:<10.4f} {result['std']:<10.4f} "
              f"{result['cv']:<10.2f} {i+1:<5}")
    
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)
    
    # 找出最佳和次佳方法
    best = all_results[0]
    second_best = all_results[1]
    
    print(f"🏆 最佳方法: {best['method']} + {best['strategy']}")
    print(f"   - 平均吞吐量: {best['mean']:.4f}")
    print(f"   - 标准差: {best['std']:.4f}")
    print(f"   - 变异系数: {best['cv']:.2f}%")
    
    print(f"\n🥈 次佳方法: {second_best['method']} + {second_best['strategy']}")
    print(f"   - 平均吞吐量: {second_best['mean']:.4f}")
    print(f"   - 标准差: {second_best['std']:.4f}")
    print(f"   - 变异系数: {second_best['cv']:.2f}%")
    
    # 计算性能提升
    improvement = ((best['mean'] - second_best['mean']) / second_best['mean']) * 100
    print(f"\n📈 性能提升: {improvement:.2f}%")
    
    # 分析稳定性
    most_stable = min(all_results, key=lambda x: x['cv'])
    least_stable = max(all_results, key=lambda x: x['cv'])
    
    print(f"\n🔒 最稳定方法: {most_stable['method']} + {most_stable['strategy']} (CV: {most_stable['cv']:.2f}%)")
    print(f"📊 最不稳定方法: {least_stable['method']} + {least_stable['strategy']} (CV: {least_stable['cv']:.2f}%)")

if __name__ == "__main__":
    # 运行完整的蒙特卡洛仿真
    results, detailed_results = test_channel_signal_integration_complete(
        num_channel_trials=20, 
        num_random_trials=50, 
        base_seed=0
    )
    
    # 创建综合可视化
    create_comprehensive_visualizations(results, detailed_results, 20, 50)
    
    # 打印统计摘要
    print_statistical_summary(results, detailed_results)
