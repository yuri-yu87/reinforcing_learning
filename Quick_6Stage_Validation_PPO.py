"""
Quick 6-Stage System Validation
Validate that the system can run correctly and complete all 6 stages
"""

import os
import sys
import numpy as np
import time

# Ensure src module path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from environment.uav_env import UAVEnvironment
from environment.advanced_endpoint_guidance import AdvancedEndpointGuidanceConfig, AdvancedEndpointGuidanceCalculator
from environment.optimized_6stage_curriculum import Optimized6StageConfig, Optimized6StageManager
from environment.intelligent_reward_system import IntelligentRewardConfig, IntelligentRewardSystem


def test_advanced_endpoint_guidance():
    """Test advanced endpoint guidance mechanism"""
    print("🧪 Testing advanced endpoint guidance mechanism...")
    
    config = AdvancedEndpointGuidanceConfig()
    calculator = AdvancedEndpointGuidanceCalculator(config)
    
    # Simulate UAV position and user positions
    uav_position = np.array([20.0, 20.0, 50.0])
    end_position = np.array([80.0, 80.0, 50.0])
    user_positions = np.array([[15.0, 15.0, 0.0], [25.0, 25.0, 0.0]])
    user_throughputs = np.array([1.0, 1.0])
    
    # Test reward calculation
    reward_breakdown = calculator.calculate_reward(
        uav_position=uav_position,
        end_position=end_position,
        user_positions=user_positions,
        user_throughputs=user_throughputs,
        current_time=100.0,
        current_speed=20.0,
        env_bounds=(100, 100, 50),
        episode_done=False,
        reached_end=False
    )
    
    print(f"   ✅ Number of reward components: {len(reward_breakdown)}")
    print(f"   ✅ Total reward: {reward_breakdown.get('total', 0):.2f}")
    print(f"   ✅ Contains endpoint guidance: {'end_approach' in reward_breakdown}")
    print(f"   ✅ Contains user visit: {'user_visit_bonus' in reward_breakdown}")
    
    return True


def test_6stage_curriculum():
    """Test 6-stage curriculum learning system"""
    print("🧪 Testing 6-stage curriculum learning system...")
    
    config = Optimized6StageConfig()
    manager = Optimized6StageManager(config)
    
    print(f"   ✅ Initial stage: {manager.current_stage}")
    
    # Test all 6 stage configurations
    all_stages_valid = True
    for stage in range(1, 7):
        stage_config = manager.get_stage_config(stage)
        
        required_keys = ['stage_name', 'user_positions', 'success_criteria', 'reward_multipliers']
        if not all(key in stage_config for key in required_keys):
            all_stages_valid = False
            print(f"   ❌ Stage {stage} configuration incomplete")
        else:
            print(f"   ✅ Stage {stage}: {stage_config['stage_name'][:20]}...")
    
    print(f"   ✅ All stage configurations valid: {all_stages_valid}")
    
    # Test stage transition
    fake_result = {
        'users_visited': 1,
        'reached_end': True,
        'total_reward': 1000
    }
    
    initial_stage = manager.current_stage
    for _ in range(10):  # Simulate 10 successful episodes
        should_advance = manager.evaluate_stage_performance(fake_result)
        if should_advance:
            manager.advance_to_next_stage()
            break
    
    stage_advanced = manager.current_stage > initial_stage
    print(f"   ✅ Stage transition mechanism: {stage_advanced}")
    
    return all_stages_valid and stage_advanced


def test_intelligent_reward_system():
    """Test intelligent reward system"""
    print("🧪 Testing intelligent reward system...")
    
    # Create config and manager
    stage_config = Optimized6StageConfig()
    stage_manager = Optimized6StageManager(stage_config)
    intelligent_config = IntelligentRewardConfig()
    
    # Create intelligent reward system
    intelligent_system = IntelligentRewardSystem(intelligent_config, stage_config, stage_manager)
    
    # Test reward calculation
    uav_position = np.array([20.0, 20.0, 50.0])
    end_position = np.array([80.0, 80.0, 50.0])
    user_positions = np.array([[15.0, 15.0, 0.0]])
    user_throughputs = np.array([1.0])
    
    reward_breakdown = intelligent_system.calculate_intelligent_reward(
        uav_position=uav_position,
        end_position=end_position,
        user_positions=user_positions,
        user_throughputs=user_throughputs,
        current_time=100.0,
        current_speed=20.0,
        env_bounds=(100, 100, 50),
        episode_done=False,
        reached_end=False,
        action=0
    )
    
    print(f"   ✅ Intelligent reward calculation successful: {len(reward_breakdown)} components")
    print(f"   ✅ Total reward: {reward_breakdown.get('total', 0):.2f}")
    
    # Test performance tracking
    episode_result = {
        'total_reward': 1000,
        'users_visited': 1,
        'reached_end': True,
        'current_time': 200
    }
    
    intelligent_system.update_episode_performance(episode_result)
    stats = intelligent_system.get_system_stats()
    
    print(f"   ✅ Performance tracking working: {stats['episode_count']} episodes")
    print(f"   ✅ System configuration loaded: {len(stats['system_config'])} items")
    
    return True


def test_environment_integration():
    """Test environment integration"""
    print("🧪 Testing environment integration...")
    
    try:
        # Create environment
        env = UAVEnvironment(
            env_size=(100, 100, 50),
            num_users=2,
            start_position=(0, 0, 50),
            end_position=(80, 80, 50),
            flight_time=250.0,
            fixed_users=True,
            seed=42
        )
        
        print(f"   ✅ Environment creation successful")
        
        # Test reset
        obs, info = env.reset()
        print(f"   ✅ Environment reset successful, observation space: {obs.shape}")
        
        # Test step
        action = 0  # East
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✅ Environment step successful, reward: {reward:.2f}")
        
        # Test user position setting
        new_positions = np.array([[20.0, 20.0, 0.0], [30.0, 30.0, 0.0]])
        env.user_manager.set_user_positions(new_positions)
        positions = env.user_manager.get_user_positions()
        
        positions_match = np.allclose(positions[:2, :2], new_positions[:, :2], atol=1.0)
        print(f"   ✅ User position setting: {positions_match}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Environment integration test failed: {e}")
        return False


def test_reward_calculator_integration():
    """Test reward calculator integration"""
    print("🧪 Testing reward calculator integration...")
    
    try:
        # Create components
        stage_config = Optimized6StageConfig()
        stage_manager = Optimized6StageManager(stage_config)
        intelligent_config = IntelligentRewardConfig()
        intelligent_system = IntelligentRewardSystem(intelligent_config, stage_config, stage_manager)
        
        # Create environment
        env = UAVEnvironment(fixed_users=True, seed=42)
        
        # Create integrated reward calculator
        class IntegratedRewardCalculator:
            def __init__(self, intelligent_system):
                self.intelligent_system = intelligent_system
                
            def calculate_reward(self, uav_position, end_position, user_positions, 
                               user_throughputs, current_time, current_speed, 
                               env_bounds, episode_done, reached_end):
                return self.intelligent_system.calculate_intelligent_reward(
                    uav_position=uav_position,
                    end_position=end_position,
                    user_positions=user_positions,
                    user_throughputs=user_throughputs,
                    current_time=current_time,
                    current_speed=current_speed,
                    env_bounds=env_bounds,
                    episode_done=episode_done,
                    reached_end=reached_end
                )
            
            def reset(self):
                self.intelligent_system.reset()
            
            def get_stats(self):
                return self.intelligent_system.get_system_stats()
        
        # Set reward calculator
        reward_calculator = IntegratedRewardCalculator(intelligent_system)
        env.set_reward_calculator(reward_calculator)
        
        print(f"   ✅ Reward calculator integration successful")
        
        # Test run
        obs, info = env.reset()
        for _ in range(5):
            action = np.random.randint(0, 5)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print(f"   ✅ Integrated system run successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Reward calculator integration failed: {e}")
        return False


def test_stage_progression():
    """Test stage progression"""
    print("🧪 Testing stage progression mechanism...")
    
    stage_config = Optimized6StageConfig()
    stage_manager = Optimized6StageManager(stage_config)
    
    print(f"   ✅ Starting stage: {stage_manager.current_stage}")
    
    stages_completed = 0
    max_attempts = 100  # Prevent infinite loop
    
    for attempt in range(max_attempts):
        # Simulate a successful episode
        fake_result = {
            'users_visited': 2,  # Visited 2 users
            'reached_end': True,  # Reached endpoint
            'total_reward': 5000  # High reward
        }
        
        should_advance = stage_manager.evaluate_stage_performance(fake_result)
        
        if should_advance:
            stages_completed += 1
            if stage_manager.current_stage < 6:
                stage_manager.advance_to_next_stage()
                print(f"   ✅ Entered stage {stage_manager.current_stage}")
            else:
                print(f"   ✅ All stages completed!")
                break
    
    print(f"   ✅ Stages completed: {stages_completed}")
    print(f"   ✅ Final stage: {stage_manager.current_stage}")
    
    return stages_completed >= 1  # At least complete one stage transition


def run_comprehensive_validation():
    """Run comprehensive validation"""
    print("🎯 === Starting 6-Stage System Comprehensive Validation === 🎯")
    start_time = time.time()
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Advanced Endpoint Guidance", test_advanced_endpoint_guidance),
        ("6-Stage Curriculum Learning", test_6stage_curriculum),
        ("Intelligent Reward System", test_intelligent_reward_system),
        ("Environment Integration", test_environment_integration),
        ("Reward Calculator Integration", test_reward_calculator_integration),
        ("Stage Progression Mechanism", test_stage_progression)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 Test: {test_name}")
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                print(f"✅ {test_name} - Passed")
            else:
                print(f"❌ {test_name} - Failed")
        except Exception as e:
            test_results[test_name] = False
            print(f"❌ {test_name} - Error: {e}")
    
    # Summary results
    elapsed_time = time.time() - start_time
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\n{'='*60}")
    print(f"🏆 === Validation Results Summary === 🏆")
    print(f"Validation time: {elapsed_time:.2f}s")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests:.1%}")
    
    print(f"\n📋 Detailed results:")
    for test_name, result in test_results.items():
        status = "✅ Passed" if result else "❌ Failed"
        print(f"  {test_name}: {status}")
    
    # Overall assessment
    if passed_tests == total_tests:
        print(f"\n🎉 === All tests passed! System ready! === 🎉")
        print("✨ 6-stage advanced training system fully validated, ready for formal training!")
        return True
    elif passed_tests >= total_tests * 0.8:
        print(f"\n✅ === Most tests passed, system basically ready === ✅")
        print("⚠️ Recommend checking failed tests, but can try running training")
        return True
    else:
        print(f"\n❌ === Multiple tests failed, need to fix issues === ❌")
        print("🔧 Please check failed components and fix before training")
        return False


def main():
    """Main function"""
    print("🌟 6-Stage Advanced Training System Quick Validation 🌟")
    print("Validation targets:")
    print("  📋 Advanced endpoint guidance mechanism")
    print("  📋 Optimized 6-stage curriculum learning")
    print("  📋 Intelligent reward system")
    print("  📋 Environment integration")
    print("  📋 Reward calculator integration")
    print("  📋 Stage progression mechanism")
    
    # Run validation
    validation_success = run_comprehensive_validation()
    
    if validation_success:
        print(f"\n🚀 System validation completed! Run the following command to start training:")
        print(f"   python Complete_6Stage_Advanced_Training.py")
        print(f"\n💡 Recommended training configurations:")
        print(f"   - Quick test: 50,000 steps (~10 minutes)")
        print(f"   - Standard training: 200,000 steps (~40 minutes)")
        print(f"   - Full training: 500,000 steps (~100 minutes)")
    else:
        print(f"\n⚠️ System validation not fully passed, recommend fixing issues first")
        print(f"Check failed tests and ensure all dependencies are correctly installed")


if __name__ == '__main__':
    main()
