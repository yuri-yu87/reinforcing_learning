# UAV-aided Telecommunication System with Reinforcement Learning

This project implements a reinforcement learning environment for optimizing UAV trajectory and transmission strategy in wireless communication systems.

## Project Structure

```
TaskF/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── uav.py              # UAV class with movement and trajectory tracking
│   │   ├── users.py            # Ground user management
│   │   └── uav_env.py          # Main RL environment
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── channel.py          # Channel model with LoS path loss
│   │   └── signal.py           # Signal processing and beamforming
│   ├── agents/                 # RL agents (to be implemented)
│   └── experiments/            # Training and evaluation scripts (to be implemented)
├── notebooks/                  # Jupyter notebooks for analysis
├── data/                       # Data storage
├── results/                    # Results and visualizations
├── requirements.txt            # Python dependencies
├── test_basic.py              # Basic component tests
├── test_environment.py        # Full environment tests
└── Design_Journal.md          # Project documentation
```

## System Model

### Environment Parameters
- **Environment Size**: 100×100×50 meters
- **UAV**: 4 antennas, fixed height at 50m
- **Users**: 2 ground users randomly distributed
- **Speed**: 10-30 m/s
- **Transmit Power**: 0.5W
- **Frequency**: 2.4 GHz
- **Path Loss Exponent**: 2.5

### Channel Model
- **Line-of-Sight (LoS)** path loss model
- **Channel Coefficient**: h = √(L₀/d^η) × h_LoS
- **SNR**: SNR = (P × |h|²) / σ²
- **Throughput**: R = B × log₂(1 + SNR)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Test the basic components:
```bash
python test_basic.py
```

3. Test the full environment (requires gymnasium):
```bash
python test_environment.py
```

## Usage

### Basic Environment Usage

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.uav_env import UAVEnvironment

# Create environment
env = UAVEnvironment(
    num_users=2,
    num_antennas=8,
    episode_length=200,
    seed=42
)

# Reset environment
obs, info = env.reset()

# Run episode
for step in range(50):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

# Get results
trajectory = env.get_trajectory()
total_throughput = env.get_total_throughput()
print(f"Total throughput: {total_throughput}")
```

### Action Space
- **Movement Direction**: [x, y] normalized direction vector
- **Speed**: Speed in m/s (10-30 m/s range)

### Observation Space
- **UAV Position**: [x, y, z] coordinates
- **Remaining Time**: Normalized remaining episode time
- **User Positions**: [x, y, z] for each user
- **Throughput History**: Last 5 throughput values

## Implementation Status

### ✅ Completed (Phase 1)
- [x] UAV class with movement and trajectory tracking
- [x] Ground user management with random distribution
- [x] Channel model with LoS path loss
- [x] Signal processing with beamforming algorithms
- [x] Main RL environment integration
- [x] Basic testing framework

### 🔄 In Progress (Phase 2)
- [ ] Baseline algorithms implementation
- [ ] Environment wrappers for normalization
- [ ] Advanced testing and validation

### 📋 Planned (Phase 3-4)
- [ ] RL algorithm implementation (PPO, SAC)
- [ ] Neural network architectures
- [ ] Training pipeline
- [ ] Experiment evaluation
- [ ] Visualization tools

## Key Features

1. **Realistic Channel Modeling**: Implements LoS path loss with configurable parameters
2. **Multi-antenna Support**: Beamforming algorithms for optimal transmission
3. **Flexible Environment**: Configurable parameters for different scenarios
4. **Comprehensive Tracking**: Trajectory, throughput, and performance metrics
5. **RL-Ready**: Compatible with gymnasium and stable-baselines3

## Testing

Run the test suite to verify all components are working:

```bash
# Test basic components
python test_basic.py

# Test full environment (requires gymnasium)
python test_environment.py
```

## Next Steps

1. **Install Dependencies**: Ensure all required packages are installed
2. **Run Tests**: Verify environment functionality
3. **Implement Baseline Algorithms**: Create comparison baselines
4. **Develop RL Agents**: Implement PPO/SAC algorithms
5. **Training Pipeline**: Set up training and evaluation framework

## Contact

For questions or issues, please refer to the Design Journal for detailed implementation notes. 