"""
参数优化分析 - 基于轨迹图和奖励分布的调参建议
目标：既能到达终点又能访问两个用户
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

def analyze_current_issues():
    """分析当前参数配置的问题"""
    
    print("🔍 === 当前配置问题分析 === 🔍")
    
    # 环境参数
    start_pos = np.array([0, 0])
    end_pos = np.array([80, 80])
    user1_pos = np.array([15, 75])
    user2_pos = np.array([75, 15])
    
    # 当前参数（用户修改后）
    current_params = {
        'user_service_radius': 30.0,
        'close_to_user_threshold': 20.0,
        'close_to_end_threshold': 30.0,
        'end_position_tolerance': 20.0,
        'B_reach_end': 3500.0,
        'B_fair_access': 1000.0,
        'B_visit_all_users': 1000.0,
        'w_user_approach': 150.0
    }
    
    print("\n📍 空间距离分析:")
    dist_start_to_user1 = np.linalg.norm(start_pos - user1_pos)
    dist_start_to_user2 = np.linalg.norm(start_pos - user2_pos)
    dist_user1_to_user2 = np.linalg.norm(user1_pos - user2_pos)
    dist_start_to_end = np.linalg.norm(start_pos - end_pos)
    
    print(f"  起点到用户1: {dist_start_to_user1:.1f}m")
    print(f"  起点到用户2: {dist_start_to_user2:.1f}m") 
    print(f"  用户1到用户2: {dist_user1_to_user2:.1f}m")
    print(f"  起点到终点: {dist_start_to_end:.1f}m")
    
    print("\n❌ 参数矛盾分析:")
    print(f"  用户服务半径: {current_params['user_service_radius']}m")
    print(f"    问题: 用户距起点76m，服务半径30m不足以覆盖")
    print(f"  用户接近阈值: {current_params['close_to_user_threshold']}m")
    print(f"    问题: 20m阈值太小，UAV很难获得接近奖励")
    print(f"  终点引导范围: {current_params['close_to_end_threshold']}m")
    print(f"    问题: 30m范围太小，无法在中途提供引导")
    
    return current_params

def propose_balanced_parameters():
    """提出平衡的参数配置"""
    
    print("\n💡 === 平衡参数配置建议 === 💡")
    
    # 建议的参数配置
    balanced_params = {
        # === 核心服务参数 ===
        'user_service_radius': 40.0,      # 适中：既能服务用户又不过大
        'close_to_user_threshold': 50.0,   # 扩大：提早给予接近奖励
        'close_to_end_threshold': 60.0,    # 适中：平衡用户访问和终点引导
        'end_position_tolerance': 15.0,    # 中等挑战：比20m严格但可达成
        
        # === 奖励权重平衡 ===
        'w_throughput_base': 100.0,
        'w_movement_bonus': 25.0,
        'w_distance_progress': 40.0,       # 增强：鼓励持续进展
        'w_user_approach': 120.0,          # 适中：平衡用户和终点引导
        'w_stagnation': 8.0,               # 增强：防止停滞
        
        # === 终端奖励策略 ===
        'B_mission_complete': 2500.0,      # 最高：完成所有任务
        'B_reach_end': 2000.0,             # 高：到达终点重要
        'B_visit_all_users': 1500.0,       # 中高：访问用户重要
        'B_fair_access': 800.0,            # 中等：公平访问
        'B_time_window': 800.0,            # 保持：时间约束
        
        # === 时间和检测参数 ===
        'user_visit_time_threshold': 1.0,  # 增加：确保真正服务用户
        'stagnation_threshold': 0.6,       # 严格：防止微小移动
        'stagnation_time_window': 2.0,     # 短：快速检测停滞
    }
    
    print("🎯 平衡参数设计理念:")
    print("  1. 🎪 渐进奖励: 接近→服务→完成，层层递进")
    print("  2. ⚖️ 平衡权重: 用户访问与终点到达并重")
    print("  3. 🎓 合理难度: 有挑战但可达成的目标")
    print("  4. 🔄 鼓励移动: 强化距离进展，惩罚停滞")
    
    print(f"\n🔧 关键参数调整:")
    for key, value in balanced_params.items():
        print(f"  {key}: {value}")
    
    return balanced_params

def create_strategy_visualization():
    """创建策略可视化"""
    
    print("\n📊 创建策略路径可视化...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 环境设置
    start_pos = np.array([0, 0])
    end_pos = np.array([80, 80])
    user1_pos = np.array([15, 75])
    user2_pos = np.array([75, 15])
    
    # === 左图：当前参数问题 ===
    ax1.set_title('❌ 当前参数配置问题', fontsize=14, fontweight='bold', color='red')
    
    # 当前参数的圆圈
    current_service_radius = 30.0
    current_end_threshold = 30.0
    current_tolerance = 20.0
    
    # 用户服务圆圈（太小）
    for i, user_pos in enumerate([user1_pos, user2_pos]):
        circle = plt.Circle(user_pos, current_service_radius, 
                          fill=False, color='purple', linestyle='--', 
                          linewidth=2, alpha=0.7)
        ax1.add_patch(circle)
        ax1.scatter(user_pos[0], user_pos[1], c='purple', s=120, marker='x')
        ax1.text(user_pos[0]+3, user_pos[1]+3, f'用户{i+1}', fontsize=10)
    
    # 终点圆圈
    end_circle = plt.Circle(end_pos, current_tolerance, 
                          fill=False, color='red', linestyle='-', linewidth=3)
    ax1.add_patch(end_circle)
    
    end_guide_circle = plt.Circle(end_pos, current_end_threshold,
                                fill=False, color='orange', linestyle=':', linewidth=2)
    ax1.add_patch(end_guide_circle)
    
    # 标记点
    ax1.scatter(*start_pos, c='green', s=150, marker='o', label='起点')
    ax1.scatter(*end_pos, c='red', s=200, marker='*', label='终点')
    
    # 问题标注
    ax1.text(45, 45, '问题分析:\n• 服务半径30m太小\n• 引导范围30m不足\n• 用户距起点76m\n• 难以平衡访问与到达', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
            fontsize=10, ha='center')
    
    ax1.set_xlim(-10, 100)
    ax1.set_ylim(-10, 100)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # === 右图：建议参数配置 ===
    ax2.set_title('✅ 建议平衡参数配置', fontsize=14, fontweight='bold', color='green')
    
    # 建议参数的圆圈
    suggested_service_radius = 40.0
    suggested_user_threshold = 50.0
    suggested_end_threshold = 60.0
    suggested_tolerance = 15.0
    
    # 用户服务圆圈（适中）
    for i, user_pos in enumerate([user1_pos, user2_pos]):
        # 服务圆圈
        service_circle = plt.Circle(user_pos, suggested_service_radius, 
                                  fill=False, color='purple', linestyle='--', 
                                  linewidth=2, alpha=0.7)
        ax2.add_patch(service_circle)
        
        # 接近奖励圆圈
        approach_circle = plt.Circle(user_pos, suggested_user_threshold, 
                                   fill=False, color='blue', linestyle=':', 
                                   linewidth=1.5, alpha=0.6)
        ax2.add_patch(approach_circle)
        
        ax2.scatter(user_pos[0], user_pos[1], c='purple', s=120, marker='x')
        ax2.text(user_pos[0]+3, user_pos[1]+3, f'用户{i+1}', fontsize=10)
    
    # 终点圆圈
    end_circle = plt.Circle(end_pos, suggested_tolerance, 
                          fill=False, color='red', linestyle='-', linewidth=3)
    ax2.add_patch(end_circle)
    
    end_guide_circle = plt.Circle(end_pos, suggested_end_threshold,
                                fill=False, color='orange', linestyle=':', linewidth=2)
    ax2.add_patch(end_guide_circle)
    
    # 标记点
    ax2.scatter(*start_pos, c='green', s=150, marker='o', label='起点')
    ax2.scatter(*end_pos, c='red', s=200, marker='*', label='终点')
    
    # 建议路径
    optimal_path = np.array([[0, 0], [15, 75], [75, 15], [80, 80]])
    ax2.plot(optimal_path[:, 0], optimal_path[:, 1], 'g--', linewidth=3, alpha=0.7, label='建议路径')
    
    # 优势标注
    ax2.text(45, 45, '优化方案:\n• 服务半径40m适中\n• 接近奖励50m范围\n• 引导范围60m充足\n• 渐进式奖励机制', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
            fontsize=10, ha='center')
    
    ax2.set_xlim(-10, 100)
    ax2.set_ylim(-10, 100)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def create_reward_structure_comparison():
    """创建奖励结构对比"""
    
    print("\n📈 创建奖励结构对比...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 当前配置 vs 建议配置
    categories = ['用户服务半径', '接近奖励阈值', '终点引导范围', '容忍度', '终点奖励/100']
    current_values = [30, 20, 30, 20, 35]  # 3500/100
    suggested_values = [40, 50, 60, 15, 20]  # 2000/100
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, current_values, width, label='当前配置', color='red', alpha=0.7)
    ax1.bar(x + width/2, suggested_values, width, label='建议配置', color='green', alpha=0.7)
    
    ax1.set_xlabel('参数类型')
    ax1.set_ylabel('数值')
    ax1.set_title('参数配置对比', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 奖励权重对比
    reward_types = ['终点到达', '用户访问', '公平访问', '任务完成']
    current_rewards = [3500, 1000, 1000, 2000]
    suggested_rewards = [2000, 1500, 800, 2500]
    
    x2 = np.arange(len(reward_types))
    
    ax2.bar(x2 - width/2, current_rewards, width, label='当前奖励', color='red', alpha=0.7)
    ax2.bar(x2 + width/2, suggested_rewards, width, label='建议奖励', color='green', alpha=0.7)
    
    ax2.set_xlabel('奖励类型')
    ax2.set_ylabel('奖励值')
    ax2.set_title('奖励权重对比', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(reward_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def provide_implementation_guide():
    """提供实施指南"""
    
    print("\n🔧 === 实施指南 === 🔧")
    
    print("📝 具体修改步骤:")
    print("1. 🎯 用户服务参数:")
    print("   user_service_radius: 30.0 → 40.0")
    print("   close_to_user_threshold: 20.0 → 50.0")
    print("   user_visit_time_threshold: 0.5 → 1.0")
    
    print("\n2. 🧭 终点引导参数:")
    print("   close_to_end_threshold: 30.0 → 60.0")
    print("   end_position_tolerance: 20.0 → 15.0")
    
    print("\n3. 💰 奖励权重调整:")
    print("   w_distance_progress: 50.0 → 40.0")
    print("   w_user_approach: 150.0 → 120.0")
    print("   w_stagnation: 10.0 → 8.0")
    
    print("\n4. 🏆 终端奖励平衡:")
    print("   B_reach_end: 3500.0 → 2000.0")
    print("   B_visit_all_users: 1000.0 → 1500.0")
    print("   B_mission_complete: 2000.0 → 2500.0")
    print("   B_fair_access: 1000.0 → 800.0")
    
    print("\n🎯 预期效果:")
    print("  ✅ UAV能够访问两个用户（40m服务半径）")
    print("  ✅ 更早获得接近奖励（50m阈值）") 
    print("  ✅ 平衡的用户-终点引导（60m vs 40m）")
    print("  ✅ 合理的精度挑战（15m容忍度）")
    print("  ✅ 鼓励完整任务（最高完成奖励）")

def analyze_trajectory_pattern():
    """分析轨迹模式"""
    
    print("\n🔄 === 轨迹模式分析 === 🔄")
    
    print("📊 从您的图表观察:")
    print("  • East/South动作主导 → UAV偏向东南方向")
    print("  • 速度恒定30m/s → 缺乏策略性调速")  
    print("  • 距离快速下降到19.8m → 可能直接冲向终点")
    print("  • 轨迹较短 → 可能因为早期终止")
    
    print("\n🎯 建议的理想轨迹:")
    print("  1. 🏁 起点(0,0) → 用户1(15,75)")
    print("  2. 👤 服务用户1 → 获得访问奖励")
    print("  3. 🔄 用户1 → 用户2(75,15)")
    print("  4. 👤 服务用户2 → 获得公平奖励")
    print("  5. 🎯 用户2 → 终点(80,80)")
    print("  6. 🏆 到达终点 → 获得完成奖励")
    
    print("\n⚙️ 实现关键:")
    print("  • 渐进奖励引导: 接近→服务→移动→完成")
    print("  • 平衡参数设置: 不偏向任何单一目标")
    print("  • 适度惩罚机制: 防止停滞但不过度")

def main():
    print("🎯 === UAV参数优化分析报告 === 🎯")
    print("目标: 既能到达终点又能访问两个用户\n")
    
    # 1. 分析当前问题
    current_params = analyze_current_issues()
    
    # 2. 提出平衡参数
    balanced_params = propose_balanced_parameters()
    
    # 3. 创建可视化
    create_strategy_visualization()
    create_reward_structure_comparison()
    
    # 4. 分析轨迹模式
    analyze_trajectory_pattern()
    
    # 5. 提供实施指南
    provide_implementation_guide()
    
    print(f"\n🎉 === 总结 === 🎉")
    print("核心思路: 平衡用户访问与终点到达，通过渐进式奖励引导完整任务")
    print("关键调整: 适中的服务半径 + 平衡的奖励权重 + 合理的挑战难度")
    print("预期效果: UAV按 起点→用户1→用户2→终点 的路径完成任务")

if __name__ == "__main__":
    main()
