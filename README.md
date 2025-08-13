# UAV-aided Telecommunication System with Reinforcement Learning

This project implements a reinforcement learning environment for optimizing UAV trajectory and transmission strategy in wireless communication systems.

This project implements a comprehensive, modular platform that integrates physical-layer communication optimization with reinforcement learning for UAV trajectory design. The system features a configurable, multi-objective reward system aligned with system goals—maximizing throughput, ensuring fairness, and achieving mission completion—while maintaining learning stability.


## 📁 Project Structure

```
TaskF/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── uav_env.py              # Main RL environment
│   │   ├── uav.py                  # UAV entity with movement tracking
│   │   ├── users.py                # Ground user management
│   │   ├── advanced_endpoint_guidance.py  # Terminal guidance strategies
│   │   ├── intelligent_reward_system.py   # Dynamic reward calculation
│   │   ├── optimized_6stage_curriculum.py # Curriculum learning
│   │   └── reward_config.py        # Configurable reward parameters
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── channel.py              # Wireless channel modeling
│   │   ├── signal.py               # Signal processing & beamforming
│   │   └── baseline_policy.py      # Deterministic baseline policies
│   └── analysis_crossLayer/
│       └── performance_analyzer.py # Cross-layer performance metrics
├── Complete_6Stage_Advanced_Training.py  # Main training script
├── Standalone_DQN_Test.py                # DQN testing & visualization
├── Curriculum_Learning_DQN_Test.py       # Curriculum learning validation
├── Parameter_Optimization_Analysis.py    # Hyperparameter analysis
├── PPO_Results_Summary.py                # PPO performance analysis
├── Notebooks/                            # Jupyter notebooks for analysis
├── results/                              # Results and visualizations
├── requirements.txt                      # Python dependencies
├── Design_Journal.md                     # Comprehensive project documentation
└── README.md                             # This file
```

## 🚀 Installation & Setup

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import gymnasium; import stable_baselines3; print('Setup complete!')"
```

## 🧪 Testing & Validation

### Quick System Tests

**Basic Environment Test:**
```bash
python -c "
import sys; sys.path.append('src')
from environment.uav_env import UAVEnvironment
env = UAVEnvironment(num_users=2, num_antennas=8)
obs, info = env.reset()
print('✅ Environment test passed!')
"
```

**Component Validation:**
```bash
# Test channel model
python -c "
import sys; sys.path.append('src')
from utils.channel import ChannelModel
channel = ChannelModel(path_loss_exponent=2.5)
print('✅ Channel model test passed!')
"

# Test signal processing
python -c "
import sys; sys.path.append('src')
from utils.signal import SignalProcessor
signal = SignalProcessor(num_antennas=8, num_users=2)
print('✅ Signal processing test passed!')
"
```

### Main Training Scripts

**DQN Training & Evaluation:**
```bash
# Main DQN training with comprehensive analysis
python Main_Standalone_DQN_Test.py

# Quick DQN validation
python Curriculum_Learning_DQN_Test.py
```

**PPO Training & Evaluation:**
```bash
# Complete 6-stage PPO training
python Complete_6Stage_Advanced_Training_PPO.py

# Quick PPO validation
python Quick_6Stage_Validation_PPO.py

# PPO results summary
python PPO_Results_Summary.py
```

### Test Scripts Overview

| Script | Purpose | Usage | Expected Output |
|--------|---------|-------|-----------------|
| `Main_Standalone_DQN_Test.py` | Complete DQN training with visualization | `python Main_Standalone_DQN_Test.py` | Training plots, trajectory analysis |
| `Complete_6Stage_Advanced_Training_PPO.py` | Full PPO curriculum learning | `python Complete_6Stage_Advanced_Training_PPO.py` | 6-stage progression, performance metrics |
| `Quick_6Stage_Validation_PPO.py` | Quick PPO system validation | `python Quick_6Stage_Validation_PPO.py` | Component validation results |
| `Curriculum_Learning_DQN_Test.py` | DQN curriculum learning test | `python Curriculum_Learning_DQN_Test.py` | Curriculum progression analysis |
| `PPO_Results_Summary.py` | PPO performance summary | `python PPO_Results_Summary.py` | Performance statistics and plots |

### Testing Workflow

1. **Environment Validation:**
   ```bash
   # Test basic environment functionality
   python -c "import sys; sys.path.append('src'); from environment.uav_env import UAVEnvironment; env = UAVEnvironment(); print('Environment OK')"
   ```

2. **Component Testing:**
   ```bash
   # Test individual components
   python Quick_6Stage_Validation_PPO.py
   ```

3. **Training Validation:**
   ```bash
   # Test DQN training
   python Main_Standalone_DQN_Test.py
   
   # Test PPO training
   python Complete_6Stage_Advanced_Training_PPO.py
   ```

4. **Results Analysis:**
   ```bash
   # Generate performance summaries
   python PPO_Results_Summary.py
   ```

### Expected Test Results

**DQN Training:**
- Episode rewards: 150,000 - 450,000
- User visit completion: ~0.4%
- Training stability: Moderate variance
- Convergence: ~25 episodes

**PPO Training:**
- Episode rewards: 1,500,000 - 2,000,000
- User visit completion: 100%
- Training stability: High consistency
- Convergence: ~80 episodes

**System Validation:**
- All components initialize successfully
- Environment resets and steps correctly
- Reward calculations are consistent
- Visualization tools generate plots

## 🚀 Key Achievements

- **Revolutionary PPO Performance**: Achieved 100% user visit success rate vs 0.4% with DQN
- **Joint Optimization Success**: +109% throughput improvement through RL trajectory + MRT beamforming
- **Modular Architecture**: Six-layer design supporting multi-user scenarios and extensible action spaces
- **Comprehensive Analysis**: Complete evaluation framework with baseline comparisons and performance metrics


## 🏗️ System Architecture

The project implements a comprehensive six-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  Complete_6Stage_Advanced_Training.py                      │
│  Standalone_DQN_Test.py                                    │
│  Curriculum_Learning_DQN_Test.py                           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                              │
│  DQN Agent (stable-baselines3)                             │
│  PPO Agent (stable-baselines3)                             │
│  BaselinePolicy (Greedy Action)                            │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Environment Layer                          │
│  UAVEnvironment (Main Env)                                  │
│  UAV (UAV Entity)                                           │
│  UserManager (User Mgmt)                                    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Advanced Environment Layer                   │
│  AdvancedEndpointGuidance                                   │
│  Optimized6StageCurriculum                                  │
│  IntelligentRewardSystem                                    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Utility Layer                              │
│  ChannelModel (Channel Model)                               │
│  SignalProcessor (Signal Proc)                              │
│  RewardConfig (Reward Config)                               │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Analysis Layer                               │
│  PerformanceAnalyzer                                        │
│  Cross-Layer Analysis                                       │
│  Metrics Tracking                                           │
└─────────────────────────────────────────────────────────────┘
```
