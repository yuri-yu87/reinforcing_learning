# ELEC9123 Design Task F (AI Optimization for UAV-aided Telecom) - Term T2, 2025

**Project Title:** Reinforcement Learning for Trajectory Design in UAV-aided Telecommunication Systems

**Author:** Yuri Yu  
**Submission File:** `z5226692_Yu_DTF_2025.zip`


## Project Directory Structure

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

## Project Timeline

This project was initiated on Wednesday of Week 8 and is scheduled for final submission and presentation on Wednesday of Week 11. The following table outlines the four-week work plan:

| Week | Phase | Objectives | Deliverables |
|------|-------|------------|--------------|
| **Week 8-9** | **Phase 1: Environment Modeling & System Design** | • Establish complete UAV communication system model<br>• Implement 3D environment and UAV/user classes<br>• Develop signal processing modules | • Environment classes (`src/environment/`)<br>• Signal processing utilities (`src/utils/`)<br>• Constraint implementation |
| **Week 9-10** | **Phase 2: RL Environment Development** | • Transform optimization problem into MDP<br>• Implement Gym environment<br>• Develop baseline algorithms | • MDP formulation<br>• Gym environment implementation<br>• Baseline algorithms (`src/agents/`) |
| **Week 10-11** | **Phase 3: RL Algorithm Implementation** | • Implement and train RL algorithms<br>• Design neural network architectures<br>• Establish training pipeline | • PPO/SAC algorithm implementation<br>• Neural network architectures<br>• Training framework (`src/experiments/`) |
| **Week 11** | **Phase 4: Experiments & Evaluation** | • Conduct comprehensive experiments<br>• Generate performance analysis<br>• Create visualizations | • Performance evaluation results<br>• Trajectory and throughput visualizations<br>• Final report and presentation |

### Detailed Implementation Plan

#### Phase 1: Environment Modeling and System Design (Week 8-9)

**Objective:** Establish complete UAV communication system model

**Key Tasks:**
1. **Environment Modeling** (`src/environment/`)
   - Implement 3D environment class (100×100×50m space)
   - Implement UAV class with position, velocity, antenna attributes
   - Implement ground user class with random distribution
   - Implement channel model (LoS path loss model)

2. **Signal Processing Module** (`src/utils/`)
   - Implement SNR calculation functions
   - Implement throughput calculation functions
   - Implement beamforming algorithms
   - Implement channel coefficient computation

3. **Constraint Implementation**
   - UAV start/end point constraints
   - Speed constraints (10-30 m/s)
   - Transmit power constraints (0.5W)
   - Boundary constraints

#### Phase 2: Reinforcement Learning Environment Development (Week 9-10)

**Objective:** Transform optimization problem into MDP and implement RL environment

**Key Tasks:**
1. **MDP Modeling** (`src/environment/`)
   - Define state space: UAV position, remaining time, user locations, historical throughput
   - Define action space: movement direction, transmit signal vectors
   - Define reward function: total throughput or throughput increment
   - Implement state transition function

2. **Gym Environment Implementation**
   - Inherit from `gym.Env` class
   - Implement `reset()`, `step()`, `render()` methods
   - Implement environment initialization and termination conditions
   - Add environment wrappers for normalization

3. **Baseline Algorithm Implementation** (`src/agents/`)
   - Implement straight-line trajectory baseline
   - Implement random trajectory baseline
   - Implement simple heuristic algorithms

#### Phase 3: Reinforcement Learning Algorithm Implementation (Week 10-11)

**Objective:** Implement and train RL algorithms

**Key Tasks:**
1. **RL Algorithm Selection** (`src/agents/`)
   - Implement PPO (recommended for continuous action space)
   - Implement SAC (alternative for continuous action space)
   - Implement DQN (if using discrete action space)

2. **Neural Network Architecture Design**
   - Design Actor network (policy network)
   - Design Critic network (value network)
   - Implement multi-input networks (handle UAV position, user locations, etc.)

3. **Training Pipeline Implementation** (`src/experiments/`)
   - Implement training loop
   - Implement experience replay buffer
   - Implement parameter update logic
   - Implement training monitoring and logging

#### Phase 4: Experiments and Evaluation (Week 11)

**Objective:** Conduct experiments and generate results

**Key Tasks:**
1. **Experimental Design** (`src/experiments/`)
   - Design different user distribution scenarios
   - Design different parameter configurations (path loss exponent, transmit power, etc.)
   - Implement multiple run averaging

2. **Performance Evaluation**
   - Compare RL algorithms with baseline algorithms
   - Analyze convergence curves
   - Visualize trajectories and throughput
   - Calculate statistical metrics

3. **Result Visualization** (`notebooks/`)
   - Trajectory visualization
   - Throughput comparison plots
   - Convergence curve plots
   - Parameter sensitivity analysis

### Technical Stack

**Core Libraries:**
- `gymnasium` - RL environment framework
- `stable-baselines3` - RL algorithm implementations
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `torch` - Deep learning (for custom networks if needed)


### Key Milestones and Risk Management

**Critical Milestones:**
- **End of Week 8:** Complete environment modeling and basic signal processing ✅
- **End of Week 9:** Complete RL environment implementation and baseline algorithms
- **End of Week 10:** Complete RL algorithm implementation and initial training
- **End of Week 11:** Complete all experiments, evaluation, and reporting

**Risk Mitigation Strategies:**
- **RL Training Convergence Issues:** Start with simple scenarios, gradually increase complexity
- **Environment Complexity:** Use pre-trained models or transfer learning
- **Computational Resources:** Implement early stopping and checkpoint saving, use cloud GPU resources (Google Colab)
- **Training Instability:** Implement proper reward scaling and normalization

### Implementation Progress

**Phase 1 Completed (Week 8-9):**
- ✅ **UAV Class** (`src/environment/uav.py`): Complete implementation with movement, trajectory tracking, and constraints
- ✅ **User Management** (`src/environment/users.py`): Ground user generation, position management, and throughput tracking
- ✅ **Channel Model** (`src/utils/channel.py`): LoS path loss model with SNR and throughput calculation
- ✅ **Signal Processing** (`src/utils/signal.py`): Beamforming algorithms (MRT, ZF) and power allocation
- ✅ **Main Environment** (`src/environment/uav_env.py`): Complete RL environment with gymnasium compatibility
- ✅ **Testing Framework**: Basic component tests and environment validation
- ✅ **Documentation**: README and comprehensive code documentation

**Key Features Implemented:**
1. **Realistic Channel Modeling**: LoS path loss with configurable parameters (η=2.5, f=2.4GHz)
2. **Multi-antenna Support**: Maximum Ratio Transmission (MRT) beamforming
3. **Flexible Environment**: Configurable UAV speed (10-30 m/s), power constraints, user distribution
4. **Comprehensive Tracking**: Trajectory history, throughput metrics, performance monitoring
5. **RL-Ready Interface**: Compatible with gymnasium and stable-baselines3

**Technical Specifications Met:**
- Environment size: 100×100×50 meters
- UAV: 4 antennas, fixed height at 50m
- Users: 2 ground users with random distribution
- Transmit power: 0.5W
- Noise power: -100 dB
- Episode length: 200-300 time steps



## 1. Introduction

With the rapid expansion of wireless connectivity and the increasing demand for reliable service in remote or disaster-affected areas, unmanned aerial vehicles (UAVs) equipped with aerial base stations (ABS) have emerged as a promising solution for providing wireless coverage. This project investigates the application of reinforcement learning (RL) to optimize the trajectory and transmission strategy of a UAV-based ABS, aiming to maximize the total data throughput delivered to ground users. The challenge lies in making real-time, intelligent decisions in a dynamic environment, which is highly relevant for next-generation adaptive communication networks.


## 2. Objectives

- To model a UAV-aided wireless communication system with multiple ground users.
- To formulate the UAV trajectory and transmit signal design as an optimization problem with the goal of maximizing the sum throughput over a flight episode.
- To reformulate the problem as a Markov Decision Process (MDP) suitable for RL-based solution.
- To implement and evaluate RL algorithms for trajectory optimization, comparing with deterministic baselines.


## 3. System Model

### 3.1 Scenario Description

- A UAV equipped with $N_t$ antennas acts as an aerial base station (ABS), serving $K$ ground users randomly distributed in a 3D environment.
- The UAV starts at $(x_0, y_0, z_h)$ and ends at $(x_L, y_L, z_h)$, flying at a fixed height $z_h$.
- The environment is a rectangular box: $x \in [x_{min}, x_{max}]$, $y \in [y_{min}, y_{max}]$, $z = z_h$.
- The UAV's total flight time is $L$ seconds, with speed $v \in [10, 30]$ m/s.

### 3.2 Signal Model

- The received signal at user $k$ at time $t$ is:
  $$ y_k(t) = h_k x_k(t) + n(t) $$
  where $h_k$ is the channel coefficient, $x_k(t)$ is the transmit signal, and $n(t)$ is AWGN noise.
- The channel is modeled as line-of-sight (LoS):
  $$ h_k = \sqrt{\frac{L_0}{d_k^\eta}} h_k^{LoS} $$
  where $d_k$ is the distance from UAV to user $k$, $\eta$ is the path loss exponent, $L_0$ is a constant, and $h_k^{LoS}$ is the LoS component.
- The SNR at user $k$:
  $$ \mathrm{SNR}_k(t) = \frac{\mathbb{E}[|h_k x_k(t)|^2]}{\mathbb{E}[|n(t)|^2]} = \frac{P}{\sigma^2} $$
- The throughput for user $k$ at time $t$:
  $$ R_k(t) = \log_2(1 + \mathrm{SNR}_k(t)) $$

  Uniform Linear Array, ULA: $\lamda/2$ classic theory

### 3.3 System Parameters

| Parameter                  | Symbol         | Value                  |
|----------------------------|---------------|------------------------|
| $x_{min}, y_{min}, z_{min}$| $x_{min}, y_{min}, z_{min}$ | 0 m |
| $x_{max}, y_{max}, z_h$    | $x_{max}, y_{max}, z_h$     | 100 m, 100 m, 50 m |
| UAV start location         | $(x_0, y_0, z_h)$ | (0, 0, 50) m         |
| UAV end location           | $(x_L, y_L, z_h)$ | (80, 80, 50) m       |
| Number of users            | $K$           | 2                     |
| Number of antennas         | $N_t$         | 4                     |
| Transmit power budget      | $P$           | 0.5 W                  |
| Noise power                | $\sigma^2$    | -100 dB                |
| Episode length             | $L$           | 200–300 s              |
| Frequency                  | $f$           | 2.4 GHz                |
| UAV speed                  | $v$           | 10–30 m/s              |
| Path loss exponent         | $\eta$        | 2.5                    |

---

## 4. Optimization Problem Formulation

### 4.1 Problem Statement

**Goal:** Maximize the sum throughput of all users over one complete UAV flight episode.

### 4.2 Decision Variables
- UAV trajectory: $\{(x_1, y_1, z_h), (x_2, y_2, z_h), ..., (x_{L-1}, y_{L-1}, z_h)\}$
- Transmit signal vectors: $\{\mathbf{w}_k(t)\}_{k=1}^K$ for $t = 0, ..., L-1$

### 4.3 Constraints
- UAV must start at $(x_0, y_0, z_h)$ and end at $(x_L, y_L, z_h)$
- UAV must not fly out of the 3D grid
- UAV must maintain constant height $z_h$
- UAV speed constraint: $\sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2} \leq v \Delta t$
- Transmit power constraint: $\|\mathbf{w}_k(t)\|^2 \leq P$

### 4.4 Mathematical Formulation

$$
\begin{align*}
\text{maximize}_{\{(x_t, y_t)\}, \{\mathbf{w}_k(t)\}} \quad & \sum_{t=0}^{L-1} \sum_{k=1}^K R_k(t) \\
\text{subject to} \quad & (x_0, y_0) = (x_{start}, y_{start}) \\
& (x_{L}, y_{L}) = (x_{end}, y_{end}) \\
& x_{min} \leq x_t \leq x_{max}, \; y_{min} \leq y_t \leq y_{max} \\
& \sqrt{(x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2} \leq v \Delta t \\
& \|\mathbf{w}_k(t)\|^2 \leq P, \; \forall k, t
\end{align*}
$$

---

## 5. Reinforcement Learning and MDP Modeling

### 5.1 Markov Decision Process (MDP) Formulation
- **State ($s_t$):** Current UAV position $(x_t, y_t, z_h)$, remaining time, user locations, previous throughput, etc.
- **Action ($a_t$):** Next movement direction (e.g., discrete steps in $x$/$y$), transmit signal/beamforming vector.
- **Reward ($r_t$):** Sum throughput at time $t$ (or increment in throughput).
- **Transition:** Environment updates UAV position and computes new state after action.
- **Episode:** Starts at $t=0$ and ends at $t=L$ (or when UAV reaches end location).

### 5.2 RL Solution Approach
- The RL agent (UAV controller) interacts with the environment, choosing actions to maximize cumulative reward (sum throughput).
- The problem can be solved using deep RL algorithms such as PPO, SAC, or DQN, depending on the action/state space.

---

## 6. Algorithm Implementation Flow

1. **Environment Initialization:**
   - Set up 3D environment and randomly place $K$ users.
   - Initialize UAV at start location.
2. **Episode Simulation:**
   - For each time step $t$:
     - Observe current state $s_t$.
     - Select action $a_t$ (move UAV, set transmit signal).
     - Update UAV position and compute channel vectors.
     - Calculate SNR and throughput for each user.
     - Compute reward $r_t$.
     - Transition to next state $s_{t+1}$.
   - Repeat until episode ends.
3. **RL Training:**
   - Use collected transitions $(s_t, a_t, r_t, s_{t+1})$ to update the RL policy.
   - Train until convergence.
4. **Evaluation:**
   - Compare RL-optimized trajectory with deterministic baselines (e.g., straight-line path).
   - Analyze throughput, convergence, and trajectory patterns.

---

## 7. Benchmark Design and Experimental Framework

### 7.1 Four Benchmark Scenarios

The project implements four distinct benchmark scenarios to comprehensively evaluate the performance of UAV trajectory and signal optimization. Each scenario represents a different combination of trajectory and beamforming strategies:

| Scenario | UAV Trajectory | Signal/Beamforming | Implementation Method | Purpose |
|----------|----------------|-------------------|---------------------|---------|
| **Benchmark 1** | Deterministic (Baseline) | Optimized (MRT/ZF/MMSE) | Straight-line path + Classical beamforming | Baseline with optimal signal processing |
| **Benchmark 2** | Deterministic (Baseline) | Randomized | Straight-line path + Random beamformers | Baseline with suboptimal signal processing |
| **Benchmark 3** | Optimized (RL) | Randomized | RL-optimized trajectory + Random beamformers | RL trajectory optimization only |
| **Benchmark 4** | Optimized (RL) | Optimized (MRT/ZF/MMSE) | RL-optimized trajectory + Classical beamforming | Full optimization (trajectory + signal) |

### 7.2 Implementation Strategy

#### 7.2.1 Trajectory Optimization
- **Baseline Trajectory**: Straight-line path from start to end position
- **Optimized Trajectory**: Reinforcement Learning algorithms (PPO/SAC/DQN)
- **Rationale**: RL is well-suited for trajectory optimization due to the high-dimensional, dynamic nature of the problem

#### 7.2.2 Signal/Beamforming Optimization
- **Classical Methods**: MRT (Maximum Ratio Transmission), ZF (Zero Forcing), MMSE (Minimum Mean Square Error)
- **Randomized Methods**: Random beamformer initialization
- **Rationale**: Classical beamforming methods provide optimal or near-optimal solutions with clear mathematical foundations, making them more suitable than RL for signal optimization

#### 7.2.3 Algorithm Selection
- **Primary RL Algorithm**: PPO (Proximal Policy Optimization) for continuous action space
- **Alternative Algorithms**: SAC (Soft Actor-Critic), DQN (Deep Q-Network) for comparison
- **Beamforming Methods**: MRT as primary method, with ZF and MMSE for comparison

### 7.3 Experimental Parameters

| Parameter                  | Value                  |
|----------------------------|------------------------|
| Environment size           | $100 \times 100 \times 50$ m |
| Number of users ($K$)      | 2                      |
| Number of antennas ($N_t$) | 4                      |
| UAV speed ($v$)            | 10–30 m/s              |
| Transmit power ($P$)       | 0.5 W                  |
| Noise power ($\sigma^2$)   | -100 dB                |
| Path loss exponent ($\eta$)| 2.5                    |
| Episode length ($L$)       | 200–300 s              |
| Frequency ($f$)            | 2.4 GHz                |
| Flight time                | 30.0 s                 |
| Time step                  | 0.1 s                  |

### 7.4 Evaluation Metrics
- **Primary Metrics:**
  - Total sum throughput over episode
  - Individual user throughput
  - Final distance to target
  - Completion time
- **Secondary Metrics:**
  - Trajectory visualization
  - Convergence curves
  - Fairness index
  - Throughput over time

### 7.5 Expected Results and Analysis

#### 7.5.1 Performance Hierarchy
Based on theoretical analysis and similar studies, the expected performance ranking (from worst to best) should be:
1. **Benchmark 2**: Baseline trajectory + Random beamforming (worst)
2. **Benchmark 1**: Baseline trajectory + Optimized beamforming
3. **Benchmark 3**: RL trajectory + Random beamforming
4. **Benchmark 4**: RL trajectory + Optimized beamforming (best)

#### 7.5.2 Key Insights to Demonstrate
- **RL Trajectory Advantage**: Benchmark 3 should outperform Benchmark 1, showing RL's ability to optimize trajectory even with suboptimal signal processing
- **Signal Optimization Impact**: Benchmark 1 should significantly outperform Benchmark 2, demonstrating the importance of proper beamforming
- **Full Optimization Benefit**: Benchmark 4 should achieve the highest throughput, showing the combined advantage of RL trajectory and optimal beamforming

#### 7.5.3 Visualization Requirements
- **Bar plots**: Sum and individual throughputs for all four benchmarks
- **Trajectory plots**: UAV paths for each benchmark scenario
- **Throughput over time**: Temporal performance comparison
- **Convergence curves**: Training progress for RL algorithms

---

## 8. Results and Discussion

- **RL vs. Baseline:** RL-optimized trajectories should yield higher sum throughput compared to straight-line or random baselines, especially when user locations are non-uniform.
- **Trajectory Patterns:** RL agent learns to hover or slow down near user clusters to maximize throughput.
- **Convergence:** Training curves should show increasing and stabilizing reward (throughput) over episodes.
- **Parameter Impact:** Analyze how path loss exponent, transmit power, and user distribution affect performance.
- **Serving All Users:** RL solution should ensure all users are served during the UAV's flight.

---

## 9. Conclusion

This project demonstrates the application of reinforcement learning to optimize UAV trajectory and transmission strategy in a wireless communication system. By modeling the problem as an MDP and leveraging deep RL algorithms, the UAV can intelligently adapt its path and transmission to maximize total throughput, outperforming deterministic baselines. The approach is scalable to more users and complex environments, providing a foundation for future research in AI-driven wireless networks.

---

## 10. References

1. A. Goldsmith, "Wireless Communications," Cambridge University Press, 2005.
2. E. Björnson et al., "Optimal Multiuser Transmit Beamforming: A Difficult Problem with a Simple Solution Structure," [Lecture Notes]. https://ieeexplore.ieee.org/abstract/document/6832894
3. Maxim Lapan, "Deep Reinforcement Learning with Python." https://github.com/PacktPublishing/Deep-Reinforcement-Learning-with-Python
4. Stable-baselines3 Documentation: https://stable-baselines3.readthedocs.io/en/master/
5. OpenAI Gym Custom Environments: https://gymnasium.farama.org/introduction/create_custom_env/

---

*Note: This report was prepared with the assistance of AI tools for technical writing and formatting. All external sources and tools used are properly cited.*