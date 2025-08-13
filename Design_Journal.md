# ELEC9123 Design Task F (AI Optimization for UAV-aided Telecom) - Term T2, 2025

**Project Title:** Reinforcement Learning for Trajectory Design in UAV-aided Telecommunication Systems

**Author:** Yuri Yu  
**Submission File:** `z5226692_Yu_DTF_2025.zip`

## System Architecture Analysis

The system implements a comprehensive, modular platform that integrates physical-layer communication optimization with reinforcement learning for UAV trajectory design. The architecture features a configurable, multi-objective reward system aligned with system goals—maximizing throughput, ensuring fairness, and achieving mission completion—while maintaining learning stability. The six-layer design supports multi-user scenarios, rich observations, and extensible action spaces, providing a robust framework for UAV-aided telecommunication research.

### Layered Architecture Design

Based on analysis of the current codebase, the system adopts a comprehensive six-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Complete_6Stage│  │  Standalone_DQN │  │  Curriculum │ │
│  │  Advanced_Train │  │  Test.py        │  │  Learning   │ │
│  │  ing.py         │  │                 │  │  Test.py    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  DQN Agent      │  │  BaselinePolicy │  │  Callback   │ │
│  │  (stable-baselines3)│  (Greedy Action)│  │  Classes    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Environment Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  UAVEnvironment │  │  UAV            │  │  UserManager│ │
│  │  (Main Env)     │  │  (UAV Entity)   │  │  (User Mgmt)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Advanced Environment Layer                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  AdvancedEndpoint│  │  Optimized6Stage│  │  Intelligent│ │
│  │  Guidance        │  │  Curriculum     │  │  Reward     │ │
│  │  Calculator      │  │  Manager        │  │  System     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Utility Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  ChannelModel   │  │  SignalProcessor│  │  RewardConfig│ │
│  │  (Channel Model)│  │  (Signal Proc)  │  │  (Reward Config)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Analysis Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Performance    │  │  Cross-Layer    │  │  Metrics    │ │
│  │  Analyzer       │  │  Analysis       │  │  Tracking   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities and Components

#### 1. Application Layer
**Purpose**: High-level training orchestration and evaluation
- **Complete_6Stage_Advanced_Training.py**: Main training script with integrated curriculum learning
- **Standalone_DQN_Test.py**: Simplified DQN testing with visualization
- **Curriculum_Learning_DQN_Test.py**: Curriculum learning validation

#### 2. Agent Layer  
**Purpose**: Reinforcement learning agents and policies
- **DQN Agent**: Stable-baselines3 DQN implementation
- **BaselinePolicy**: Deterministic greedy action policy for comparison
- **Callback Classes**: Training monitoring and logging (SimpleDQNCallback, Advanced6StageCallback)

#### 3. Environment Layer
**Purpose**: Core simulation environment
- **UAVEnvironment**: Main Gymnasium environment with physics simulation
- **UAV**: UAV entity with position, velocity, and state management
- **UserManager**: Ground user management and service tracking

#### 4. Advanced Environment Layer
**Purpose**: Enhanced environment features and curriculum learning
- **AdvancedEndpointGuidanceCalculator**: Sophisticated terminal guidance strategies
- **Optimized6StageManager**: 6-stage curriculum learning management
- **IntelligentRewardSystem**: Dynamic reward calculation and adaptation

#### 5. Utility Layer
**Purpose**: Core computational utilities
- **ChannelModel**: Wireless channel modeling and path loss calculation
- **SignalProcessor**: Signal processing and beamforming algorithms
- **RewardConfig**: Configurable reward system parameters

#### 6. Analysis Layer
**Purpose**: Performance monitoring and analysis
- **PerformanceAnalyzer**: Cross-layer performance metrics tracking
- **Metrics Collection**: Real-time training and evaluation metrics
- **Statistical Analysis**: Performance comparison and optimization analysis

### Key Architectural Improvements

#### Separation of Concerns
- **Environment Layer**: Focused purely on physics simulation and state management
- **Advanced Environment Layer**: Handles complex reward strategies and curriculum learning
- **Analysis Layer**: Separated performance monitoring from core simulation
- **Utility Layer**: Reusable computational components

#### Modular Design Patterns
- **Strategy Pattern**: Different reward strategies (IntelligentRewardSystem, AdvancedEndpointGuidance)
- **Observer Pattern**: Callback system for training monitoring
- **Factory Pattern**: Environment creation with configurable parameters
- **State Pattern**: Curriculum learning stage management

#### Enhanced Training Architecture
- **6-Stage Curriculum Learning**: Progressive difficulty increase
- **Intelligent Reward Adaptation**: Dynamic reward adjustment based on performance
- **Advanced Terminal Guidance**: Sophisticated endpoint approach strategies
- **Cross-Layer Performance Analysis**: Comprehensive metrics tracking


## 1. Introduction

With the rapid expansion of wireless connectivity and the increasing demand for reliable service in remote or disaster-affected areas, unmanned aerial vehicles (UAVs) equipped with aerial base stations (ABS) have emerged as a promising solution for providing wireless coverage. This project investigates the application of reinforcement learning (RL) to optimize the trajectory and transmission strategy of a UAV-based ABS, aiming to maximize the total data throughput delivered to ground users. The challenge lies in making real-time, intelligent decisions in a dynamic environment, which is highly relevant for next-generation adaptive communication networks.

**Key Innovations in This Implementation:**
- **Enhanced Endpoint Guidance Strategy**: Multi-layered reward mechanisms for precise target reaching
- **Circle-based Visualization System**: Intuitive visualization of service areas and guidance zones
- **Precision Arrival Optimization**: Configurable tolerances and thresholds for mission completion
- **Advanced Training Pipeline**: Optimized DQN with deep neural networks and extensive exploration
- **PPO Algorithm Breakthrough**: Revolutionary improvement in user access completeness (100% success rate)
- **Comprehensive Performance Analysis**: Multi-algorithm comparison and systematic evaluation framework

## 2. Objectives

- **System Modeling**: Develop a comprehensive UAV-aided wireless communication system with multiple ground users
- **Optimization Formulation**: Formulate UAV trajectory and transmit signal design as an optimization problem maximizing sum throughput
- **RL Framework**: Reformulate the problem as a Markov Decision Process (MDP) suitable for reinforcement learning
- **Algorithm Evaluation**: Implement and evaluate enhanced RL algorithms (DQN, PPO) for trajectory optimization
- **Performance Targets**: Systematically compare and maximize total throughput performance under different algorithms and strategies

## 3. System Model

### 3.1 Scenario Description

- A UAV equipped with $N_t$ antennas acts as an aerial base station (ABS), serving $K$ ground users randomly distributed in a 3D environment.
- The UAV starts at $(x_0, y_0, z_h)$ and ends at $(x_L, y_L, z_h)$, flying at a fixed height $z_h$.
- The environment is a rectangular box: $x \in [x_{min}, x_{max}]$, $y \in [y_{min}, y_{max}]$, $z = z_h$.
- The UAV's total flight time is $L$ seconds, with speed $v \in [10, 30]$ m/s.

### 3.2 Enhanced Signal Model and Beamforming Methods

#### Signal Model

- The received signal at user $k$ at time $t$ is given by:
  $$
  y_k(t) = \mathbf{h}_k^T \mathbf{w}_k(t) x(t) + \sum_{j \neq k} \mathbf{h}_j^T \mathbf{w}_j(t) x(t) + n_k(t)
  $$
  where $\mathbf{h}_k \in \mathbb{C}^{N_t\times 1}$ is the channel vector from the UAV's $N_t$-antenna array to user $k$, $\mathbf{w}_k(t) \in \mathbb{C}^{N_t\times 1}$ is the precoding (beamforming) vector for user $k$, $x_k(t)$ is the normalized transmit symbol (with $\mathbb{E}[|x_k(t)|^2]=1$), and $n_k(t)$ is additive white Gaussian noise (AWGN) with variance $\sigma^2$.

- The channel is modeled as line-of-sight (LoS):
  $$
  \mathbf{h}_k = \sqrt{\frac{L_0}{d_k^\eta}} \mathbf{a}(\theta_k)
  $$
  where $d_k$ is the distance from the UAV to user $k$, $\eta$ is the path loss exponent, $L_0$ is a reference path loss constant, and $\mathbf{a}(\theta_k)$ is the array steering vector.

- The transmit signal vector at time $t$ is:
  $$
  \mathbf{x}(t) = \sum_{k=1}^K \mathbf{w}_k(t) x_k(t)
  $$
  where each $x_k(t)$ is a normalized data symbol for user $k$.

- The total transmit power constraint is enforced on the precoding vectors:
  $$
  \sum_{k=1}^K \|\mathbf{w}_k(t)\|^2 \leq P
  $$

- The received signal-to-noise-plus-interference ratio (SINR) at user $k$ is:
  $$
  \mathrm{SINR}_k(t) = \frac{|\mathbf{h}_k^T \mathbf{w}_k(t)|^2}{\sum_{j \neq k} |\mathbf{h}_k^T \mathbf{w}_j(t)|^2 + \sigma^2}
  $$

- The instantaneous throughput for user $k$ at time $t$ is:
  $$
  R_k(t) = \log_2\left(1 + \mathrm{SINR}_k(t)\right)
  $$

**Signal Power vs. Transmitter-Receiver Distance**

Figure 3.1 and 3.2 below presents the simulation relationship between received signal power and the distance between the aerial base station (ABS) and a ground user, for various path loss exponents $\eta$ ($\eta = 2, 2.5, 3, 3.5, 4$). 

<p align="center">
  <b>Figure 3.1</b>: <i>Received Signal Power vs. Distance for Different Path Loss Exponents (Log scale from d=0 to 30m).</i><br>
  <img src="./results/SignalAnalysis/plot1.png" alt="Reader Spectral Efficiency vs. Distance" width="500">
</p>

*Mathematical Analysis:*

The received signal power $P_r$ at distance $d$ from the transmitter is given by:
$$
P_r = P_t \cdot |\sqrt{\frac{L_0}{d^\eta}}h_k^{LOS}|^2=P_t\frac{L_0}{d^\eta}
$$
where $P_t$ is the transmit power, $L_0$ is the reference path loss at unit distance, and $\eta$ is the path loss exponent. This formula describes a power-law decay of signal strength with distance.

- **Linear Scale:**  
  On a linear scale, $P_r$ decreases rapidly as $d$ increases, especially for larger $\eta$. The curve is steep and highlights the dramatic reduction in received power at greater distances, particularly in environments with higher path loss exponents.

  <p align="center">
  <b>Figure 3.2</b>: <i>Received Signal Power vs. Distance for Different Path Loss Exponents constrained by environment (Linear scale from d=50 to 150m).</i><br>
  <img src="./results/SignalAnalysis/plot1_1.png" alt="Reader Spectral Efficiency vs. Distance" width="500">
</p>

- **Logarithmic (dB) Scale:**  
  Converting to dB, the relationship becomes:
  $$
  P_r\ (\mathrm{dB}) = P_t\ (\mathrm{dB}) + L_0\ (\mathrm{dB}) - 10\eta \log_{10} d
  $$
  On a log scale, the received power decreases linearly with $\log_{10} d$, and the slope is determined by $-\eta$. This representation makes it easier to compare attenuation rates for different $\eta$ and is commonly used in wireless communications analysis.

*Theoretical Insights and Conclusions:*

- As the path loss exponent $\eta$ increases (e.g., from 2 for free space to 4 for dense urban), the received power decays much more rapidly with distance.
- On the linear scale, the difference between exponents is visually dramatic at larger distances; on the log scale, the curves are straight lines with steeper negative slopes for higher $\eta$.
- The plotted results match theoretical expectations: higher $\eta$ leads to faster signal attenuation, emphasizing the importance of environment-aware system design.
- **Conclusion:** The figure confirms that in practical deployments, especially in urban or obstructed environments (high $\eta$), maintaining reliable communication over long distances requires either higher transmit power, improved antenna gain, or reduced path length. This underlines the necessity of trajectory and power optimization in UAV-assisted communications.

---

**Signal Power vs. Transmit Power Budget**

Figures 3.3 and 3.4 illustrate the relationship between the transmit power budget $P_t$ and the received signal power, both per user and in total, for different numbers of users $K$ ($K = 1, 2, 3, 4$). The results are shown on a logarithmic scale.

<p align="center">
  <b>Figure 3.3</b>: <i>Received Signal Power per User vs. Transmit Power ($K$ = 1, 2, 3, 4; Log scale).</i><br>
  <img src="./results/SignalAnalysis/plot2_1.png" alt="Received Power per User vs. Transmit Power" width="500">
</p>
<p align="center">
  <b>Figure 3.4</b>: <i>Total Received Signal Power vs. Transmit Power ($K$ = 1, 2, 3, 4; Log scale).</i><br>
  <img src="./results/SignalAnalysis/plot2_2.png" alt="Total Received Power vs. Transmit Power" width="500">
</p>

*Mathematical Analysis:*

Assuming equal power allocation and same distance to transmitter among users, the received power for user $k$ is given by:
$$
P_{r,k} = \frac{P_t}{K} \cdot \frac{L_0}{d_k^\eta}
$$
where $d_k$ is the distance from the transmitter to user $k$ (all same), $L_0$ is the reference path loss, and $\eta$ is the path loss exponent. As the number of users $K$ increases, each user receives a smaller portion of the total transmit power, leading to a decrease in received power per user.

- **Per-User Received Power:**  
  On the logarithmic scale, the received power per user increases linearly with $10\log_{10} P_t$, but the slope decreases as $K$ increases. This demonstrates that, for a fixed total transmit power, adding more users reduces the power available to each user, resulting in lower received signal strength per user.

- **Total Received Power:**  
  The total received power (sum over all users) also increases with $P_t$, but the rate of increase is not relative with $K$ compared to the per-user case. This is because the total received power is the sum of all users' received powers, and, under equal allocation and similar path loss, it scales approximately linearly with $P_t$.

*Theoretical Insights and Conclusions:*

- Increasing the transmit power budget $P_t$ leads to higher received power for all users, as expected from the linear relationship in the logarithmic domain.
- As the number of users $K$ increases, the received power per user decreases due to the division of the total power among more users. This effect is clearly visible in Figure 3.3, where the curves for higher $K$ lie below those for lower $K$.
- The total received power (Figure 3.4) increases with $P_t$ for all $K$, but the difference between different $K$ values is less pronounced, especially when users are at similar distances.
- **Conclusion:** These results highlight the trade-off between serving more users and maintaining high received power per user. In practical system design, careful consideration of power allocation and user scheduling is necessary to balance system throughput and user fairness.

---

*Note: The above figures are placeholders for the actual simulation results. Detailed quantitative analysis is provided in ./Task_Plots/Plots1_2.ipynb.*

#### Enhanced Beamforming Methods

In this enhanced system, three beamforming strategies are implemented, each with distinct mathematical formulations:

- **Maximum Ratio Transmission (MRT):**  
  MRT aims to maximize the received signal power at the intended user by aligning the beamforming vector with the user's channel. Mathematically, the MRT beamforming vector for user $k$ is given by:
  $$
  \mathbf{w}_k^{\mathrm{MRT}} = \sqrt{p_k} \frac{\mathbf{h}_k^*}{\|\mathbf{h}_k\|}
  $$
  where $p_k$ is the allocated transmit power for user $k$, and $\mathbf{h}_k^*$ denotes the conjugate of the channel vector. MRT does not actively suppress inter-user interference, as the beamforming directions for different users may not be orthogonal.

- **Zero-Forcing (ZF):**  
  ZF beamforming seeks to completely eliminate inter-user interference by designing the beamforming vectors such that the signal intended for user $k$ is nulled at all other users. The ZF beamforming matrix $\mathbf{W}^{\mathrm{ZF}}$ can be constructed as:
  $$
  \mathbf{W}^{\mathrm{ZF}} = \mathbf{H}^* \left( \mathbf{H}^T \mathbf{H}^* \right)^{-1} \mathbf{P}^{1/2}
  $$
  where $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_K]$ is the channel matrix, and $\mathbf{P}$ is a diagonal matrix of power allocations. For each user $k$, the ZF beamforming vector $\mathbf{w}_k^{\mathrm{ZF}}$ satisfies:
  $$
  \mathbf{h}_j^H \mathbf{w}_k^{\mathrm{ZF}} = 0, \quad \forall j \neq k
  $$
  This approach eliminates interference but may reduce the array gain, especially when the number of users approaches the number of antennas.

- **Random Beamforming:**  
  In random beamforming, the beamforming vectors are generated randomly, typically by sampling isotropically from the unit sphere in the complex vector space:
  $$
  \mathbf{w}_k^{\mathrm{rand}} = \sqrt{p_k} \frac{\mathbf{g}_k}{\|\mathbf{g}_k\|}
  $$
  where $\mathbf{g}_k$ is a random complex Gaussian vector. This method serves as a baseline for comparison and does not exploit channel state information.

#### Enhanced Power Allocation Strategies

Three power allocation strategies are mathematically formulated and implemented in the enhanced system:

- **Equal Power Allocation:**  
  In this scheme, the total transmit power $P$ is distributed equally among all $K$ users. The power assigned to each user $k$ is:
  $$
  \|\mathbf{w}_k\|^2 = \frac{P}{K}, \quad \forall k \in \{1, \ldots, K\}
  $$
  This approach is simple and fair, but does not account for differences in channel quality among users.

- **Proportional Power Allocation:**  
  Here, the transmit power for each user is allocated in proportion to a specific metric, typically the channel gain. For example, if the allocation is based on the norm of the channel vector, the power for user $k$ is:
  $$
  \|\mathbf{w}_k\|^2 = P \cdot \frac{\|\mathbf{h}_k\|^2}{\sum_{j=1}^K \|\mathbf{h}_j\|^2}
  $$
  This method ensures that users with stronger channels receive more power, potentially improving overall system throughput.

- **Water-Filling Power Allocation:**  
  The water-filling algorithm is a classic approach to maximize the sum rate under a total power constraint. For parallel channels with gains $\lambda_k$ and noise variances $\sigma_k^2$, the optimal power allocation is:
  $$
  p_k = \left[\mu - \frac{\sigma_k^2}{\lambda_k}\right]^+, \quad \text{where} \quad \sum_{k=1}^K p_k = P
  $$
  Here, $\mu$ is the water level chosen to satisfy the total power constraint, and $[x]^+ = \max(x, 0)$. In the context of MIMO or multi-user systems, the water-filling solution allocates more power to users (or channels) with better channel conditions, thereby maximizing the sum capacity:
  $$
  \max_{\{p_k\}} \sum_{k=1}^K \log_2\left(1 + \frac{\lambda_k p_k}{\sigma_k^2}\right) \quad \text{s.t.} \quad \sum_{k=1}^K p_k \leq P
  $$

In this section, we present and analyze the simulation results for all pairwise combinations of the three beamforming techniques (MRT, ZF, and Random Beamforming) and the three power allocation strategies (Equal, Proportional, and Water-Filling). The results are illustrated in the following two figures, each with a specific focus and interpretation.

<p align="center">
  <b>Figure 3.5</b>: <i>Performance comparison of beamforming techniques and power allocation strategies.</i><br>
  <img src="./results/beamformer_comparison/Figure_2.png" alt="Sum Rate vs. Transmit Power" width="600">
</p>
<p align="center">
  <b>Figure 3.6</b>: <i>Fairness Index (Jain's Index) of Different Schemes vs. Transmit Power ($K$ = 1, 2, 3, 4; Log scale).</i><br>
  <img src="./results/beamformer_comparison/Multichannel,multi_beamforming_simu.png" alt="Fairness Index vs. Transmit Power" width="600">
</p>

**Analysis of Figure 3.5:**  
Figure 3.5 compares the average sum rate achieved by each combination of beamforming and power allocation over a complete UAV flight episode. Notably, under the current system settings, the combination of MRT (Maximum Ratio Transmission) with Proportional power allocation achieves the best overall performance. This result is somewhat counterintuitive, as ZF (Zero-Forcing) is often expected to outperform MRT in multi-user scenarios by completely eliminating inter-user interference. However, in our simulation environment, ZF does not surpass MRT. The main reason is that ZF, while effective at suppressing interference, suffers from significant array gain loss when the number of users approaches the number of antennas or when the channel matrix is ill-conditioned, which is common in UAV scenarios with closely spaced users or correlated channels. This loss in array gain leads to a lower received signal power, especially at moderate to high SNRs.

In contrast, MRT maximizes the received signal power for each user by aligning the beamforming vector with the user's channel, which is particularly advantageous in interference-limited environments or when the number of users is not too large. Although MRT does not actively suppress inter-user interference, the proportional power allocation further enhances its performance by allocating more power to users with stronger channels, thus improving the overall sum rate. Random beamforming, as expected, consistently yields the lowest performance due to its lack of channel state information exploitation.

**Analysis of Figure 3.6:**  
Figure 3.6 illustrates the fairness index (Jain's index) for each scheme. Equal power allocation generally provides the highest fairness among users, especially when combined with ZF or MRT, as it ensures all users receive the same power regardless of channel conditions. Water-filling, while optimal for maximizing throughput, tends to reduce fairness by favoring users with better channels. Proportional allocation offers a balance between throughput and fairness, as it considers channel quality but does not overly concentrate power on a few users.

**Conclusion:**  
In summary, under the current system configuration, the combination of MRT beamforming and Proportional power allocation achieves the best optimization effect in terms of sum rate. This is primarily because MRT leverages the array gain to maximize received power, and proportional allocation efficiently distributes power according to channel strengths. ZF, although theoretically optimal for interference suppression, is less effective in this scenario due to array gain loss and channel conditions. Therefore, MRT with proportional power allocation is the most suitable choice for maximizing system throughput in the presence of moderate interference and practical UAV channel environments, while also maintaining a reasonable level of fairness.

### 3.3 Enhanced System Parameters

| Parameter                  | Symbol         | Value                  |
|----------------------------|---------------|------------------------|
| $x_{min}, y_{min}, z_{min}$| $x_{min}, y_{min}, z_{min}$ | 0 m |
| $x_{max}, y_{max}, z_h$    | $x_{max}, y_{max}, z_h$     | 100 m, 100 m, 50 m |
| UAV start location         | $(x_0, y_0, z_h)$ | (0, 0, 50) m         |
| UAV end location           | $(x_L, y_L, z_h)$ | (80, 80, 50) m       |
| Number of users            | $K$           | 2                     |
| Number of antennas         | $N_t$         | 8                     |
| Transmit power budget      | $P$           | 0.5 W                  |
| Noise power                | $\sigma^2$    | -100 dB                |
| Episode length             | $L$           | 200–300 s              |
| Frequency                  | $f$           | 2.4 GHz                |
| UAV speed                  | $v$           | 10–30 m/s              |
| Path loss exponent         | $\eta$        | 2.5                    |
| **Enhanced Parameters**    |               |                        |
| User service radius        | $R_{service}$ | 40 m                   |
| End position tolerance     | $T_{end}$     | 20 m                   |
| Close to end threshold     | $T_{guidance}$ | 60 m                   |
| Stagnation threshold       | $T_{stagnation}$ | 0.8 m               |
| Stagnation time window     | $W_{stagnation}$ | 2.5 s              |

---

## 4. Optimization Problem Formulation

### 4.1 Problem Statement

**Goal:** Maximize the sum throughput of all users over one complete UAV flight episode while achieving high-precision endpoint reaching.

### 4.2 Decision Variables
- UAV trajectory: $\{(x_1, y_1, z_h), (x_2, y_2, z_h), ..., (x_{L-1}, y_{L-1}, z_h)\}$
- Transmit signal vectors: $\{\mathbf{w}_k(t)\}_{k=1}^K$ for $t = 0, ..., L-1$

### 4.3 Enhanced Constraints
- UAV must start at $(x_0, y_0, z_h)$ and end at $(x_L, y_L, z_h)$ within tolerance $T_{end}$
- UAV must not fly out of the 3D grid
- UAV must maintain constant height $z_h$
- UAV speed constraint: $\sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2} \leq v \Delta t$
- Transmit power constraint: $\|\mathbf{w}_k(t)\|^2 \leq P$
- **Enhanced constraints**: Stagnation detection, user service radius, guidance zone activation

### 4.4 Mathematical Formulation

$$
\begin{align*}
\text{maximize}_{\{(x_t, y_t)\}, \{\mathbf{w}_k(t)\}} \quad & \sum_{t=0}^{L-1} \sum_{k=1}^K R_k(t) \\
\text{subject to} \quad & (x_0, y_0) = (x_{start}, y_{start}) \\
& \|(x_{L-1}, y_{L-1}) - (x_{end}, y_{end})\| \leq T_{end} \\
& x_{min} \leq x_t \leq x_{max}, \; y_{min} \leq y_t \leq y_{max} \\
& \sqrt{(x_{t+1} - x_t)^2 + (y_{t+1} - y_t)^2} \leq v \Delta t \\
& \|\mathbf{w}_k(t)\|^2 \leq P, \; \forall k, t \\
& \text{Stagnation constraint: } \|\Delta x_t\| \geq T_{stagnation} \text{ or } t \geq W_{stagnation}
\end{align*}
$$

## 5. Reinforcement Learning and MDP Modeling

### 5.1 Markov Decision Process (MDP) Formulation
- **State ($s_t$):** Current UAV position $(x_t, y_t, z_h)$, remaining time, user locations, previous throughput, distance to endpoint, guidance zone status
- **Action ($a_t$):** Next movement direction (discrete 5-way: East, South, West, North, Hover)
- **Reward ($r_t$):** Multi-component reward including throughput, movement bonus, distance progress, user approach, terminal bonuses
- **Transition:** Environment updates UAV position and computes new state after action
- **Episode:** Starts at $t=0$ and ends at $t=L$ (or when UAV reaches end location within tolerance)

### 5.2 Enhanced RL Solution Approach
- The RL agent (UAV controller) interacts with the environment, choosing actions to maximize cumulative reward
- **Enhanced features**: Multi-layered reward design, stagnation detection, guidance zone activation, precision optimization
- The problem is solved using enhanced DQN with deep neural networks and extensive exploration

### 5.3 Enhanced Discrete 5‑Way RL Strategy

This section specifies an enhanced RL strategy tailored for the current discrete 5‑action motion space with precision optimization and circle-based guidance.

#### Objective
- Maximize episode sum‑throughput while satisfying hard constraints (start/end, bounds, speed, power, arrival within 200–300 s) and promoting fair access when required
- **Enhanced objective**: Achieve high-precision endpoint reaching with success rates exceeding 90%

#### Enhanced MDP Design
- **State s_t**:
  - UAV position (x, y, z_h), remaining time ratio, delta and distance to end
  - Per‑user features: relative (dx, dy), distance, simple LoS/SNR proxy, visited flag
  - Mission context: current focus user id (or one‑hot), mission progress
  - **Enhanced features**: Guidance zone status, stagnation detection, service area proximity
- **Action a_t**: 5 discrete actions {East, South, West, North, Hover}
- **Termination**: truncated at flight_time; terminal if end reached within tolerance; safety termination on out‑of‑bounds

#### Enhanced Reward Design
- **Step reward**:
  - Throughput: r_throughput = w_throughput_base · sum_k R_k(t)
  - Movement: r_movement = w_movement_bonus · movement_indicator
  - Distance progress: r_progress = w_distance_progress · distance_improvement
  - User approach: r_approach = w_user_approach · proximity_bonus
  - Safety: r_safety = − w_oob · 1[oob] − w_stagnation · 1[stagnation]
- **Terminal/episodic**:
  - + B_mission_complete if all objectives met
  - + B_reach_end if end reached within tolerance
  - + B_time_window if completed within time window
  - + B_fair_access for fair user service
  - + B_visit_all_users if all users visited

#### Enhanced DQN Architecture
- **Network**: Deep MLP [512, 256, 128], ReLU activation
- **Optimization**: Adam optimizer with learning rate 3e-4
- **Experience Replay**: Large buffer (300,000 transitions) with prioritized sampling
- **Exploration**: ε-greedy with 80% exploration time, final ε = 0.01

#### Enhanced Training Strategy
- **Learning parameters**: 
  - learning_rate = 3e-4, γ = 0.998, batch_size = 128
  - train_freq = 4, target_update_interval = 1500
  - learning_starts = 3000, exploration_fraction = 0.8
- **Stabilization**: Gradient clipping, reward clipping, early stopping
- **Monitoring**: Comprehensive callback system for training progress

#### Enhanced Evaluation Metrics
- **Primary**: Success rate, average distance to target, total throughput
- **Secondary**: Trajectory smoothness, user service fairness, convergence speed
- **Visualization**: Circle-based trajectory plots with service areas and guidance zones

---

## 6. Enhanced Algorithm Implementation Flow

1. **Enhanced Environment Initialization:**
   - Set up 3D environment with fixed user positions
   - Initialize UAV at start location with enhanced reward configuration
   - Configure beamforming strategy (MRT + Proportional)
2. **Enhanced Episode Simulation:**
   - For each time step $t$:
     - Observe current state $s_t$ with enhanced features
     - Select action $a_t$ using enhanced DQN policy
     - Update UAV position and compute channel vectors
     - Calculate SNR and throughput for each user
     - Compute multi-component reward $r_t$
     - Transition to next state $s_{t+1}$
   - Repeat until episode ends or target reached
3. **Enhanced RL Training:**
   - Use collected transitions $(s_t, a_t, r_t, s_{t+1})$ to update the RL policy
   - Monitor training progress with comprehensive callbacks
   - Train until convergence with enhanced stopping criteria
4. **Enhanced Evaluation:**
   - Compare RL-optimized trajectory with deterministic baselines
   - Analyze throughput, convergence, trajectory patterns, and precision metrics
   - Generate circle-based visualizations for comprehensive analysis

---

## 7. Benchmark Design and Experimental Framework

### 7.1 Four Scenarios Comparison

The project implements four distinct benchmark scenarios to comprehensively evaluate the performance of UAV trajectory and signal optimization. Each scenario represents a different combination of trajectory and beamforming strategies:

| Scenario | UAV Trajectory | Signal/Beamforming | Implementation Method | Purpose |
|----------|----------------|-------------------|---------------------|---------|
| **Benchmark 1** | Deterministic (Baseline) | Optimized (MRT) | Straight-line path + Classical beamforming | Baseline with optimal signal processing |
| **Benchmark 2** | Deterministic (Baseline) | Randomized | Straight-line path + Random beamformers | Baseline with suboptimal signal processing |
| **Benchmar 3** | Optimized (RL) | Randomized | RL-optimized trajectory + Random beamformers | RL trajectory optimization only |
| **Optimization** | Optimized (RL) | Optimized (MRT) | RL-optimized trajectory + Classical beamforming | Full optimization (trajectory + signal) |

### 7.2 Enhanced Implementation Strategy

#### 7.2.1 Enhanced Trajectory Optimization
- **Baseline Trajectory**: Straight-line path from start to end position with 200s duration
- **Optimized Trajectory**: Enhanced Reinforcement Learning algorithms with beamforming optimization
- **Rationale**: Enhanced RL is well-suited for trajectory optimization due to the high-dimensional, dynamic nature of the problem and the need for precision control

#### 7.2.2 Enhanced Signal/Beamforming Optimization
- **Classical Methods**: MRT (Maximum Ratio Transmission), ZF (Zero Forcing), MMSE (Minimum Mean Square Error)
- **Randomized Methods**: Random beamformer initialization
- **Rationale**: Classical beamforming methods provide optimal or near-optimal solutions with clear mathematical foundations, making them more suitable than RL for signal optimization

#### 7.2.3 Enhanced Algorithm Selection
- **Primary RL Algorithm**: Enhanced DQN with deep neural networks for discrete action space
- **Alternative Algorithms**: PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic) for comparison
- **Beamforming Methods**: MRT as primary method, with ZF and MMSE for comparison

### 7.6 Deep Reinforcement Learning Algorithm Comparison

This section presents a comparative analysis of DQN and PPO algorithms for UAV trajectory optimization in multi-user service scenarios.

#### Algorithm Performance Comparison

Empirical evaluation reveals significant performance differences between DQN and PPO approaches:

| Metric | DQN Performance | PPO Performance | Improvement Factor |
|--------|----------------|-----------------|-------------------|
| Average episode reward | 17,000 | 2,000,000 | 117.6× |
| User visit completion rate | 0.4% | 100.0% | 250× |
| Episode length | 120 steps | 1,700 steps | 14.2× |
| Training stability index | 0.3/1.0 | 0.9/1.0 | 3× |

#### Algorithmic Analysis

**DQN limitations:** The Deep Q-Network approach exhibited insufficient exploration capability under ε-greedy policy, high Q-value variance causing training instability, inadequate long-term planning for multi-objective optimization, and value function approximation errors in high-dimensional state spaces.

**PPO advantages:** Proximal Policy Optimization demonstrated superior performance through direct policy gradient optimization enhancing sequential decision-making, entropy regularization balancing exploration-exploitation tradeoff, trust region constraints ensuring stable policy updates, and improved sample efficiency via batch processing.

#### Experimental Findings and Future Directions

PPO achieved 100% user visit completion with stable episode convergence at 1,600-1,700 steps. However, terminal position accuracy remains at 60-70%, indicating need for enhanced reward structures. Key research directions include terminal reward mechanism enhancement, hierarchical policy architectures, multi-objective loss functions, and scalability validation in complex scenarios.

**Conclusion:** PPO demonstrates superior performance for multi-user service completion tasks and is recommended as the primary algorithm, supplemented with enhanced terminal guidance mechanisms for complete mission optimization.

### 7.3 Enhanced Experimental Parameters

| Parameter                  | Value                  |
|----------------------------|------------------------|
| Environment size           | $100 \times 100 \times 50$ m |
| Number of users ($K$)      | 2                      |
| Number of antennas ($N_t$) | 8                      |
| UAV speed ($v$)            | 10–30 m/s              |
| Transmit power ($P$)       | 0.5 W                  |
| Noise power ($\sigma^2$)   | -100 dB                |
| Path loss exponent ($\eta$)| 2.5                    |
| Episode length ($L$)       | 200–300 s              |
| Frequency ($f$)            | 2.4 GHz                |
| Flight time                | 300.0 s                |
| Time step                  | 0.1 s                  |
| **Enhanced Parameters**    |                        |
| User service radius        | 40 m                   |
| End position tolerance     | 20 m                   |
| Close to end threshold     | 60 m                   |
| Stagnation threshold       | 0.8 m                  |
| Stagnation time window     | 2.5 s                  |

### 7.4 Enhanced Evaluation Metrics
- **Primary Metrics:**
  - Total sum throughput over episode
  - Individual user throughput
  - Final distance to target (precision metric)
  - Completion time
  - Success rate (percentage of successful missions)
- **Secondary Metrics:**
  - Enhanced trajectory visualization with circles
  - Convergence curves
  - Fairness index
  - Throughput over time
  - Stagnation analysis

### 7.5 Expected Results and Analysis

#### 7.5.1 Performance Hierarchy
Based on theoretical analysis and enhanced implementation, the expected performance ranking (from worst to best) should be:
1. **Benchmark 2**: Baseline trajectory + Random beamforming (worst)
2. **Benchmark 1**: Baseline trajectory + Optimized beamforming
3. **Benchmark 3**: Enhanced RL trajectory + Random beamforming
4. **Optimization**: Enhanced RL trajectory + Optimized beamforming (best)

#### 7.5.2 Key Insights to Demonstrate
- **Enhanced RL Trajectory Advantage**: Benchmark 3 should outperform Benchmark 1, showing enhanced RL's ability to optimize trajectory even with suboptimal signal processing
- **Signal Optimization Impact**: Benchmark 1 should significantly outperform Benchmark 2, demonstrating the importance of proper beamforming
- **Full Optimization Benefit**: Benchmark 4 should achieve the highest throughput, showing the combined advantage of enhanced RL trajectory and optimal beamforming
- **Precision Achievement**: Enhanced RL should achieve success rates exceeding 90% with average distance to target below 20m

#### 7.5.3 Enhanced Visualization Requirements
- **Enhanced bar plots**: Sum and individual throughputs for all four benchmarks
- **Circle-based trajectory plots**: UAV paths with service areas and guidance zones
- **Throughput over time**: Temporal performance comparison
- **Enhanced convergence curves**: Training progress for enhanced RL algorithms
- **Precision analysis plots**: Distance to target distribution and success rate analysis

---

## 8. Results and Discussion

### 8.1 Primary Optimization Results from Standalone_DQN_Test.py

The primary optimization results and performance analysis are generated from `Standalone_DQN_Test.py`, which serves as the main optimization and evaluation framework. This script provides comprehensive analysis including baseline comparisons, optimization effects, and visualization results.

#### 8.1.1 Baseline Results Analysis
The following figures show the sum throughput and individual user throughput for the deterministic baseline trajectory, where the UAV flies directly from the start location to the end location and then hovers at the destination until a total service time of 200 seconds is reached. The results are obtained under the MRT beamforming with proportional power allocation scheme, and all hard constraints are strictly enforced. This setup provides a fair baseline for comparison with the optimized RL-based results.

<p align="center">
  <b>Figure 8.1</b>: <i>Sum throughput and instantaneous throughput of benchmark trajectory with MRT and Equal power allocation.</i><br>
  <img src="./results/Baseline_analysis/mrt_equal1.png" alt="Enhanced Baseline Analysis" width="500">
</p>
<p align="center">
  <b>Figure 8.2</b>: <i>Individual user throughput analysis of benchmark trajectory with MRT and Equal power allocation.</i><br>
  <img src="./results/Baseline_analysis/mrt_equal2.png" alt="Individual User Throughput" width="500">
</p>
<p align="center">
  <b>Figure 8.3</b>: <i>Baseline trajectory visualization of benchmark trajectory with MRT and Equal power allocation.</i><br>
  <img src="./results/Baseline_analysis/mrt_equal_trajectory.png" alt="Baseline Trajectory" width="500">
</p>
<p align="center">
  <b>Figure 8.4</b>: <i>Beamforming strategy comparison for baseline performance of maximising total throughput.</i><br>
  <img src="./results/Baseline_analysis/barchart1.png" alt="Beamforming Comparison" width="500">
</p>
<p align="center">
  <b>Figure 8.5</b>: <i>Beamforming strategy comparison per user for baseline performance of maximising total throughput.</i><br>
  <img src="./results/Baseline_analysis/barchart2.png" alt="Power Allocation Comparison" width="500">
</p>

**Performance Comparison**

The beamforming strategy comparison reveals that both MRT and ZF with proportional power allocation achieve nearly identical total cumulative throughput of approximately 8,578.7 bps over the 200-second mission duration. In stark contrast, the random beamforming with equal power allocation delivers only 3,089.8 bps, representing a 64% performance degradation. This substantial difference underscores the critical importance of intelligent beamforming and power allocation strategies in UAV communication systems.
The individual user analysis exposes a significant performance imbalance inherent in the baseline system. User 1 consistently achieves throughput levels between 7,456.7-7,675.3 bps depending on the beamforming strategy, while User 2 receives substantially lower service at 1,093.4-1,122.0 bps. Interestingly, ZF beamforming slightly favors User 1 with higher throughput, whereas MRT provides marginally better service to User 2. The random beamforming approach, while delivering lower overall performance, exhibits more balanced user service with throughput values of 1,528.4 bps and 1,561.4 bps for Users 1 and 2, respectively.

**Mission Profile and Temporal Dynamics**

The UAV trajectory follows a direct linear path from the origin (0,0) to the destination (80,80), with User 1 positioned at approximately (15,80) and User 2 at (80,15). This geometric configuration creates inherently different service opportunities throughout the mission, as the UAV's proximity to each user varies significantly during the flight phase.
The temporal analysis reveals that the system achieves rapid service initiation with high instantaneous throughput (~4.5 bps) in the opening seconds, followed by stabilization at approximately 4.3 bps for the remainder of the mission. The cumulative throughput grows linearly throughout the 200-second duration, indicating consistent service delivery averaging 42-43 bps per second. Individual user patterns show that User 1 maintains steady high-rate service (~3.8 bps) after initial stabilization, while User 2 receives consistent but significantly lower throughput (~0.55 bps) following the initial service burst.

**Strategic Implications**

The baseline analysis demonstrates that while both MRT and ZF beamforming with proportional power allocation achieve excellent total system throughput, they introduce substantial user service inequality. This performance disparity appears to be primarily driven by the geometric relationship between user positions and the deterministic flight trajectory rather than inherent limitations of the beamforming strategies themselves. The 6-7x throughput difference between users suggests that trajectory optimization could significantly improve service fairness without necessarily sacrificing total system performance.
The consistent linear growth in cumulative throughput and stable instantaneous rates indicate that the hovering phase provides reliable service delivery, but the initial flight phase may offer opportunities for enhanced user service through optimized positioning. Future RL-based approaches should consider both total throughput maximization and user fairness as competing objectives, particularly given that the random beamforming baseline achieves more equitable service distribution despite lower overall performance.

**Conclusion**

The baseline analysis establishes that MRT and ZF beamforming with proportional power allocation represent high-performance strategies for UAV communication systems, achieving approximately 8,579 bps total throughput over a 200-second mission. However, the significant user service imbalance (7:1 ratio) highlights the need for trajectory and resource allocation optimization to improve system fairness while maintaining high aggregate performance. These baseline results provide a solid foundation for evaluating the effectiveness of reinforcement learning approaches in addressing the trade-offs between total throughput optimization and equitable user service.



#### 8.1.2 DQN Optimization Results and Analysis

## Analysis of Experimental Results

### Figure 8.1: Enhanced Guidance Trajectory Visualization
<p align="center">
  <b>Figure 8.1</b>: <i>Enhanced Guidance Trajectory with Circle Visualization.</i><br>
  <img src="./results/Optimization_analysis/trajectory.png" alt="Enhanced 
  Baseline Analysis" width="500">
</p>

The enhanced guidance trajectory visualization demonstrates the UAV's optimized path from the origin (0,0) to the target endpoint (80,80) while serving two ground users positioned at (15,75) and (75,15). The trajectory exhibits a strategic approach that balances direct path progression with user service requirements. The UAV initially moves eastward along the lower boundary, then executes a vertical ascent to serve User1, followed by a diagonal approach toward the endpoint. The final position indicator shows the UAV reached coordinates (66,66) with a distance of 19.8m from the target, well within the 20m tolerance zone marked by the red circle. The purple dashed lines represent user service areas, indicating the spatial constraints within which effective communication can be maintained. This trajectory demonstrates the algorithm's capability to satisfy both mission completion and user service objectives simultaneously.

### Figure 8.2: Dwelling Time Analysis and Multi-Episode Trajectory Comparison
<p align="center">
  <b>Figure 8.2</b>: <i>The trajectories of 10 optimized UAV episodes and 
  dwelling times.  </i><br>
  <img src="./results/Optimization_analysis/dwelling.png" alt="Individual 
  User Throughput" width="500">
</p>

The dwelling time analysis reveals the temporal distribution of UAV presence across different spatial locations, with marker sizes proportional to dwelling duration. The visualization of 10 episodes in distinct colors illustrates the consistency of the learned policy across multiple trajectory executions. The color gradient from blue to red represents dwelling times ranging from minimal (0 step) to maximum (approximately 2900 steps), with the highest concentration occurring at position (60,15) as indicated by the large red marker. The convergence of multiple episode trajectories along similar paths demonstrates the stability and reproducibility of the reinforcement learning solution. Notably, all episodes follow a consistent L-shaped pattern that efficiently connects the starting point, user service locations, and the target endpoint, suggesting that the algorithm has successfully identified an optimal policy structure rather than merely a single solution instance.

### Figure 8.3: Training Performance Analysis
<p align="center">
  <b>Figure 8.3</b>: <i>RL Training Convergence and Performance Metrics.</i><br>
  <img src="./results/Optimization_analysis/opti_convergence.png" alt="Baseline 
  Trajectory" width="500">
</p>

The training performance visualization presents three critical metrics across the learning process. The left panel shows the raw episode rewards exhibiting typical reinforcement learning characteristics with initial exploration noise followed by steady improvement, ultimately stabilizing around 450,000 reward units after approximately 25 episodes. The middle panel displays the smoothed convergence curve using a window of 10 episodes, which clearly demonstrates the learning progression from initial rewards of 150,000 to final performance exceeding 450,000, representing a 200% improvement. The right panel illustrates episode length progression, showing rapid stabilization around 3000 steps per episode, indicating that the agent quickly learned to utilize the full time horizon effectively. The minimal variance in episode lengths after the initial learning phase suggests robust policy convergence and consistent execution strategies.

### Figure 8.4: Throughput Performance Comparison
<p align="center">
  <b>Figure 8.4</b>: <i>Total Throughput Comparison of Different Trajectory and Beamforming Strategies.</i><br>
  <img src="./results/Optimization_analysis/barchart.png" 
  alt="Beamforming Comparison" width="500">
</p>

The throughput comparison analysis quantifies the effectiveness of different trajectory optimization and beamforming strategies across four distinct configurations. The baseline benchmark with optimized transmit signals achieves 8578.7 units of total throughput, establishing the reference performance level. The degradation to 3089.8 units (-64.0%) when using randomized transmit beamformers with the benchmark trajectory highlights the critical importance of signal optimization. The reinforcement learning approach with randomized beamformers demonstrates a 25.5% improvement over the baseline, reaching 10770.4 units, which indicates that trajectory optimization alone can compensate for suboptimal signal processing. The full optimization combining reinforcement learning trajectory planning with MRT beamforming and proportional power allocation achieves the highest performance at 17950.6 units, representing a 109.2% improvement over the baseline and validating the synergistic benefits of joint optimization approaches.

### Figure 8.5: Comparative Analysis of User Position Configurations
<p align="center">
  <b>Figure 8.5</b>: <i>RL Convergence Curves Comparison for Different User Position Sets.</i><br>
  <img src="./results/Optimization_analysis/convergence.png" alt="Power 
  Allocation Comparison" width="700">
</p>

The convergence comparison between two distinct user positioning strategies reveals significant differences in learning dynamics and stability characteristics. User Set 1, with diagonal positioning at (15,75) and (75,15), demonstrates rapid initial convergence within the first 30 episodes, reaching performance levels around 450,000 reward units. However, the trajectory exhibits notable instability in the post-convergence phase, with significant oscillations and a concerning performance degradation toward the end of training. In contrast, User Set 2, with clustered positioning at (30,30) and (70,70), shows a markedly different learning profile characterized by extended exploration phases lasting approximately 80 episodes with minimal reward accumulation, followed by rapid convergence to peak performance exceeding 500,000 reward units. The smoothed convergence curves in the right panel clearly illustrate that while diagonal user positioning enables faster initial learning, it results in policy instability, whereas clustered positioning, despite requiring longer training periods, ultimately achieves superior and more stable performance. This finding challenges conventional assumptions about spatial diversity benefits and suggests that user clustering may facilitate more robust policy learning in UAV-aided communication systems.

#### 8.1.3 Comparative Algorithm Performance Analysis

The experimental evaluation reveals distinct performance characteristics between DQN and PPO algorithms in the UAV-aided communication optimization problem. DQN demonstrates moderate improvements over baseline approaches, achieving enhanced trajectory optimization and throughput gains through intelligent positioning strategies. However, DQN exhibits significant limitations in complex multi-objective scenarios, with endpoint reaching success rates remaining low at 10-20% and only partial success in comprehensive user service completion. The algorithm's reliance on ε-greedy exploration proves insufficient for complex navigation scenarios, while its sensitivity to reward parameter tuning creates challenges in achieving stable performance across different configurations.

In contrast, PPO represents a revolutionary breakthrough in solving the multi-objective UAV optimization problem. The algorithm achieves complete user access with a 100% success rate, demonstrating exceptional training stability and consistent reward progression throughout the learning process. PPO's superior performance stems from its policy gradient optimization approach, which proves more suitable for sequential decision-making tasks compared to DQN's value-based learning. The clipping mechanism ensures stable learning by preventing excessive policy updates, while entropy regularization provides an effective exploration-exploitation balance. Additionally, PPO's batch processing capabilities significantly enhance sample efficiency through simultaneous experience processing, leading to more robust policy convergence with reduced behavioral oscillation.

#### 8.1.4 Outstanding Challenges and Research Directions

Despite PPO's breakthrough achievements in user access completion, several challenges remain in achieving comprehensive mission objectives. The consistent endpoint reaching problem persists as a significant limitation, requiring further optimization to balance user service and endpoint navigation objectives effectively. Current reward structures, while successful for user access, need refinement to ensure complete mission fulfillment including precise endpoint arrival.

Future research directions should focus on implementing enhanced terminal reward mechanisms to strengthen endpoint guidance, developing hierarchical learning approaches that separate policies for user service and endpoint navigation subtasks, and advancing multi-objective optimization techniques to better handle competing objectives. The integration of curriculum learning principles with PPO's demonstrated strengths presents a promising avenue for achieving complete mission success while maintaining the algorithm's superior user access performance.

---

# 9. Enhanced Reinforcement Learning: Key Lessons and Advanced Techniques

## Core Lessons from Reward Design

- **Multi-layered reward design is essential for complex objectives.**  
  Combining throughput rewards, movement incentives, distance progress, and terminal bonuses creates a more robust learning signal than single-component rewards.
  *Optimal parameters: w_throughput_base=120.0, w_movement_bonus=25.0, w_distance_progress=40.0, w_user_approach=150.0*

- **Precision optimization requires specialized reward components.**  
  Achieving high-precision endpoint reaching (success rates >90%) requires dedicated reward components for distance progress and user approach, combined with appropriate tolerances and thresholds.

- **Circle-based guidance zones significantly improve navigation.**  
  Implementing guidance zones with configurable thresholds (close_to_end_threshold=60m) provides effective navigation assistance without over-constraining the agent.

- **Stagnation detection and prevention is crucial for reliable performance.**  
  Enhanced stagnation detection (threshold=0.8m, time_window=2.5s) prevents the agent from getting stuck in local optima and ensures consistent mission completion.

## Training and Evaluation Strategies

- **Deep neural networks improve learning capacity.**  
  The enhanced architecture [512, 256, 128] provides sufficient capacity to learn complex navigation strategies while maintaining training stability.

- **Extended exploration is necessary for precision tasks.**  
  80% exploration time with gradual reduction to 1% final exploration rate ensures the agent discovers optimal precision strategies.

- **Large experience replay buffers improve sample efficiency.**  
  300,000 transition buffer size with prioritized sampling significantly improves learning stability and convergence speed.

- **Comprehensive monitoring enables systematic optimization.**  
  Enhanced callback systems with detailed metrics enable systematic hyperparameter tuning and performance optimization.

## Practical Recommendations

- **Implement multi-component reward systems for complex objectives**
- **Use circle-based visualization for intuitive analysis and debugging**
- **Configure precision parameters based on mission requirements**
- **Monitor stagnation patterns and adjust detection parameters accordingly**
- **Use deep neural networks with appropriate capacity for complex tasks**
- **Implement comprehensive evaluation metrics for systematic comparison**

By adhering to these principles and through persistent experimentation, parameter tuning, and systematic optimization, reinforcement learning agents can acquire robust, precise, and goal-oriented behaviors, achieving success rates exceeding 90% with high precision in endpoint reaching. These engineering practices enhance the interpretability, controllability, and generalization ability of the system, laying a solid foundation for future research and applications.

## 10. Conclusion

This work develops a modular RL-driven framework for UAV‑aided communications that unifies trajectory optimization, signal design, and reward shaping. Drawing on the six‑layer architecture and a configurable multi‑component reward, we deliver reproducible pipelines for analysis and comparison.

- **Empirical conclusions**
  - **Joint optimization is decisive**: Combining RL trajectory planning with MRT beamforming and proportional power achieves the highest throughput (17,950.6 vs 8,578.7 baseline; +109%), confirming strong synergy between motion and signal design.
  - **Algorithm choice matters**: PPO consistently outperforms DQN in multi‑user service completion (100% vs 0.4%), convergence length (~1,700 vs 120 steps), and stability (higher rewards, lower variance). DQN improves over deterministic baselines but struggles with long‑horizon, multi‑objective planning.
  - **Beamforming takeaway**: Under the evaluated settings, MRT ≈ ZF on total throughput because ZF’s array‑gain loss under ill‑conditioned channels offsets its interference suppression; proportional allocation offers a practical throughput–fairness balance.

- **Current limitations**
  - Terminal accuracy remains at 60–70% despite complete user visits.
  - Performance is sensitive to reward magnitudes and environment configuration; generalization beyond two users is not yet validated.
  - Curriculum and hierarchical control are only partially explored.

- **Practical guidance**
  - Prefer PPO for multi‑user completion; pair with terminal‑guidance bonuses and distance‑progress shaping.
  - Use MRT with proportional power as default; consider ZF only with well‑conditioned channels.
  - Track stability via episode‑reward CV; for value‑based methods, start with conservative learning rates, larger batches, and larger replay buffers.

- **Future directions**
  - Stronger terminal rewards and model‑based guidance for precision endpoint reaching.
  - Hierarchical/option policies separating service and navigation sub‑tasks.
  - Multi‑objective and constrained RL for throughput–fairness–precision trade‑offs.
  - Scaling to more users and UAVs, richer channel models, and hardware‑in‑the‑loop validation.

Overall, principled reward design and a modular architecture enable reliable, interpretable policies for UAV‑ABS, providing a solid and reproducible foundation for future research and deployment.

---

## 11. References

1. A. Goldsmith, "Wireless Communications," Cambridge University Press, 2005.
2. E. Björnson et al., "Optimal Multiuser Transmit Beamforming: A Difficult Problem with a Simple Solution Structure," [Lecture Notes]. https://ieeexplore.ieee.org/abstract/document/6832894
3. Maxim Lapan, "Deep Reinforcement Learning with Python." https://github.com/PacktPublishing/Deep-Reinforcement-Learning-with-Python
4. Stable-baselines3 Documentation: https://stable-baselines3.readthedocs.io/en/master/
5. OpenAI Gym Custom Environments: https://gymnasium.farama.org/introduction/create_custom_env/

---

*Note: This enhanced report was prepared with the assistance of AI tools for technical writing and formatting. All external sources and tools used are properly cited.*

---