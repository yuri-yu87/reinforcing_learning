import numpy as np
from typing import List, Tuple, Optional


class BeamformingAlgorithm:
    """
    Beamforming algorithms for UAV multi-antenna transmission.
    """

    @staticmethod
    def maximum_ratio_transmission(channel_coeffs: np.ndarray,
                                   power_constraint: float = 0.5) -> np.ndarray:
        """
        Maximum Ratio Transmission (MRT) beamforming for Uniform Linear Array (ULA).
        Power constraint is handled here.
        """
        num_users = len(channel_coeffs)
        num_antennas = len(channel_coeffs[0])
        beamforming_vectors = np.zeros((num_users, num_antennas), dtype=complex)

        # print(f"MRT Debug: num_users={num_users}, num_antennas={num_antennas}")
        # print(f"MRT Debug: power_constraint={power_constraint}")

        # Calculate total power for normalization
        total_power = 0.0

        for i, h in enumerate(channel_coeffs):
            h_norm = np.linalg.norm(h)
            # print(f"MRT Debug: User {i+1}, channel norm = {h_norm}")
            
            if h_norm > 1e-10:  # 更严格的阈值检查
                # MRT beamforming: w = h* / ||h||
                # This maximizes the signal power for each user
                w = np.conj(h) / h_norm
                # print(f"MRT Debug: User {i+1}, w norm before = {np.linalg.norm(w)}")
            else:
                # If channel is zero, use uniform beamforming
                w = np.ones(num_antennas, dtype=complex) / np.sqrt(num_antennas)
                # print(f"MRT Debug: User {i+1}, using uniform beamforming")
            
            beamforming_vectors[i] = w
            w_power = np.linalg.norm(w) ** 2
            total_power += w_power
            # print(f"MRT Debug: User {i+1}, w power = {w_power}")
        
        # print(f"MRT Debug: total_power before normalization = {total_power}")
        
        # Normalize to satisfy power constraint
        if total_power > 1e-10:
            scale_factor = np.sqrt(power_constraint / total_power)
            # print(f"MRT Debug: scale_factor = {scale_factor}")
            beamforming_vectors *= scale_factor
            
            # 验证最终功率
            final_power = np.sum([np.linalg.norm(vec)**2 for vec in beamforming_vectors])
            # print(f"MRT Debug: final total power = {final_power}")
        else:
            print("MRT Debug: Warning - total_power is too small!")

        return beamforming_vectors

    @staticmethod
    def zero_forcing_beamforming(channel_matrix: np.ndarray,
                                 power_constraint: float = 0.5) -> np.ndarray:
        """
        Zero Forcing (ZF) beamforming.
        
        Args:
            channel_matrix: Channel matrix H where H[i,j] is channel from antenna i to user j
            power_constraint: Total power constraint
            
        Returns:
            np.ndarray: Beamforming vectors with shape (num_users, num_antennas)
            
        Power constraint is handled here.
        """
        H = channel_matrix  # Shape: (num_antennas, num_users)
        num_antennas, num_users = H.shape
        
        # print(f"ZF Debug: Channel matrix shape: {H.shape}")
        # print(f"ZF Debug: num_antennas={num_antennas}, num_users={num_users}")
        
        # Check if ZF is feasible
        if num_antennas < num_users:
            # print(f"ZF Warning: Insufficient antennas ({num_antennas}) for users ({num_users})")
            # print("ZF Warning: Falling back to MRT-like approach")
            # Fallback to MRT-like approach when ZF is not feasible
            W = np.zeros((num_users, num_antennas), dtype=complex)
            for i in range(num_users):
                h_i = H[:, i]  # Channel for user i
                h_norm = np.linalg.norm(h_i)
                if h_norm > 1e-10:
                    W[i, :] = np.conj(h_i) / h_norm
                else:
                    W[i, :] = np.ones(num_antennas, dtype=complex) / np.sqrt(num_antennas)
        else:
            # Standard ZF: W = H^H * (H * H^H)^(-1)
            H_H = np.conj(H.T)  # Shape: (num_users, num_antennas)
            
            # Calculate H * H^H
            HH_H = H @ H_H  # Shape: (num_antennas, num_antennas)
            
            # Check condition number and singularity
            cond_num = np.linalg.cond(HH_H)
            # print(f"ZF Debug: Condition number of H*H^H: {cond_num}")
            
            if cond_num < 1e12:
                try:
                    # Standard inversion
                    HH_H_inv = np.linalg.inv(HH_H)
                    W_transpose = H_H @ HH_H_inv  # Shape: (num_users, num_antennas)
                    W = W_transpose
                    # print("ZF Debug: Using standard matrix inversion")
                except np.linalg.LinAlgError:
                    # print("ZF Warning: Matrix inversion failed, using pseudo-inverse")
                    HH_H_pinv = np.linalg.pinv(HH_H)
                    W = H_H @ HH_H_pinv
            else:
                # print("ZF Warning: Ill-conditioned matrix, using pseudo-inverse")
                # Use pseudo-inverse for ill-conditioned matrix
                HH_H_pinv = np.linalg.pinv(HH_H)
                W = H_H @ HH_H_pinv
        
        # print(f"ZF Debug: W shape before normalization: {W.shape}")
        
        # Normalize to satisfy power constraint
        total_power = np.sum(np.linalg.norm(W, axis=1) ** 2)  # Sum over users
        # print(f"ZF Debug: Total power before normalization: {total_power}")
        
        if total_power > 1e-10:
            scale_factor = np.sqrt(power_constraint / total_power)
            W = W * scale_factor
            # print(f"ZF Debug: Scale factor: {scale_factor}")
        else:
            # print("ZF Warning: Total power is too small, using uniform allocation")
            W = np.ones((num_users, num_antennas), dtype=complex) / np.sqrt(num_antennas * num_users)
        
        final_power = np.sum(np.linalg.norm(W, axis=1) ** 2)
        # print(f"ZF Debug: Final total power: {final_power}")
        
        return W  # Shape: (num_users, num_antennas)
    
    @staticmethod
    def equal_power_allocation(num_users: int, power_constraint: float = 0.5) -> np.ndarray:
        """
        Equal power allocation among users (relative, normalized to sum to 1).
        """
        return np.full(num_users, 1.0 / num_users)


class SignalProcessor:
    """
    Signal processing utilities for UAV communication system.
    Implements beamforming and SINR calculation for multi-user scenarios.
    """

    def __init__(self, num_antennas: int = 8):
        self.num_antennas = num_antennas
        self.beamforming = BeamformingAlgorithm()

    def calculate_effective_channel(self,
                                   channel_coeffs: np.ndarray,
                                   beamforming_vectors: np.ndarray) -> np.ndarray:
        num_users = len(channel_coeffs)
        effective_channels = np.zeros(num_users, dtype=complex)
        for i, h in enumerate(channel_coeffs):
            if i < len(beamforming_vectors):
                w = beamforming_vectors[i]
                # Effective channel: h_eff = h^H * w
                h_eff = np.dot(np.conj(h), w)
                effective_channels[i] = h_eff
        return effective_channels

    def calculate_interference_matrix(self,
                                      channel_coeffs: np.ndarray,
                                      beamforming_vectors: np.ndarray) -> np.ndarray:
        num_users = len(channel_coeffs)
        interference_matrix = np.zeros((num_users, num_users))
        for i in range(num_users):
            for j in range(num_users):
                if i != j and j < len(beamforming_vectors):
                    h_i = channel_coeffs[i]  # Channel to user i
                    w_j = beamforming_vectors[j]  # Beamforming for user j
                    # Interference: |h_i^H * w_j|^2
                    interference = np.abs(np.dot(np.conj(h_i), w_j)) ** 2
                    interference_matrix[i, j] = interference
        return interference_matrix

    def calculate_interference(self,
                              channel_coeffs: np.ndarray,
                              beamforming_vectors: np.ndarray,
                              user_index: int) -> float:
        interference_power = 0.0
        num_users = len(channel_coeffs)
        for j in range(num_users):
            if j != user_index and j < len(beamforming_vectors):
                h_i = channel_coeffs[user_index]  # Channel to user i
                w_j = beamforming_vectors[j]  # Beamforming for user j
                # Interference: |h_i^H * w_j|^2
                interference = np.abs(np.dot(np.conj(h_i), w_j)) ** 2
                interference_power += interference
        return interference_power

    def calculate_sinr(self,
                       effective_channel_power: float,
                       interference_power: float,
                       noise_power: float) -> float:
        if interference_power + noise_power > 0:
            sinr = effective_channel_power / (interference_power + noise_power)
        else:
            sinr = 0.0
        return sinr

    def calculate_sinr_for_all_users(self,
                                     effective_channels: np.ndarray,
                                     interference_matrix: np.ndarray,
                                     power_allocation: np.ndarray,
                                     noise_power: float) -> np.ndarray:
        num_users = len(effective_channels)
        sinr_values = np.zeros(num_users)
        for i in range(num_users):
            desired_power = power_allocation[i] * np.abs(effective_channels[i]) ** 2
            interference_power = np.sum(interference_matrix[i, :] * power_allocation)
            if interference_power + noise_power > 0:
                sinr_values[i] = desired_power / (interference_power + noise_power)
            else:
                sinr_values[i] = 0.0
        return sinr_values

    def calculate_throughput_for_all_users(self,
                                           sinr_values: np.ndarray) -> np.ndarray:
        """
        Calculate the throughput for all users.
        Throughput = log2(1 + SINR)
        Args:
            sinr_values (np.ndarray): Array of SINR values for each user.
        Returns:
            np.ndarray: Throughput values for each user.
        """
        # Ensure SINR values are non-negative
        # print(f"Start calculating throughput for all users")
        # print(f"sinr_values: {sinr_values}")
        sinr_values = np.maximum(sinr_values, 0)
        # Directly compute throughput without a loop
        return np.log2(1 + sinr_values)

    def optimize_power_allocation(self,
                                 channel_coeffs: np.ndarray,
                                 power_constraint: float = 0.5,
                                 method: str = 'proportional') -> np.ndarray:
        """
        Optimize power allocation among users.
        Now only returns relative allocation (normalized to sum to 1).
        Args:
            channel_coeffs: Channel coefficients for each user
            power_constraint: Total power constraint
            method: Optimization method ('equal', 'water_filling', 'proportional')
            
        Returns:
            np.ndarray: Optimal power allocation
        """
        num_users = len(channel_coeffs)
        if method == 'equal':
            allocation = np.full(num_users, 1.0 / num_users)
        elif method == 'proportional':
            channel_gains = np.abs(channel_coeffs) ** 2
            total_gain = np.sum(channel_gains)
            if total_gain > 0:
                allocation = channel_gains / total_gain
            else:
                allocation = np.full(num_users, 1.0 / num_users)
        elif method == 'water_filling':
            channel_gains = np.abs(channel_coeffs) ** 2
            sorted_indices = np.argsort(channel_gains)[::-1]
            allocation = np.zeros(num_users)
            remaining = 1.0
            for i in range(num_users):
                if remaining <= 0:
                    break
                idx = sorted_indices[i]
                if channel_gains[idx] > 0:
                    mu = remaining / (num_users - i)
                    power = max(0, mu - 1 / channel_gains[idx])
                    allocation[idx] = power
                    remaining -= power
            if np.sum(allocation) == 0:
                allocation = np.full(num_users, 1.0 / num_users)
            else:
                allocation = allocation / np.sum(allocation)
        else:
            raise ValueError(f"Unknown power allocation method: {method}")
        return allocation

    def optimize_power_allocation_joint(self,
                                        channel_coeffs: np.ndarray,
                                        beamforming_vectors: np.ndarray,
                                        total_power_constraint: float) -> np.ndarray:
        """
        Joint optimization of power allocation considering beamforming vectors.
        Now only returns relative allocation (normalized to sum to 1).
        """
        num_users = len(channel_coeffs)
        effective_gains = []
        for i in range(num_users):
            if i < len(beamforming_vectors):
                h_eff = np.dot(np.conj(channel_coeffs[i]), beamforming_vectors[i])
                effective_gains.append(np.abs(h_eff) ** 2)
            else:
                effective_gains.append(0.0)
        effective_gains = np.array(effective_gains)
        if np.sum(effective_gains) == 0:
            return np.full(num_users, 1.0 / num_users)
        sorted_indices = np.argsort(effective_gains)[::-1]
        allocation = np.zeros(num_users)
        remaining = 1.0
        for i in range(num_users):
            if remaining <= 0:
                break
            idx = sorted_indices[i]
            if effective_gains[idx] > 0:
                mu = remaining / (num_users - i)
                power = max(0, mu - 1 / effective_gains[idx])
                allocation[idx] = power
                remaining -= power
        if np.sum(allocation) == 0:
            allocation = np.full(num_users, 1.0 / num_users)
        else:
            allocation = allocation / np.sum(allocation)
        return allocation

    def calculate_fairness_index(self, throughput_values: np.ndarray) -> float:
        # Jain's fairness index: 1 is most fair, 1/n is least fair (n = number of users)
        # Formula: (sum(x_i))^2 / (n * sum(x_i^2)), where x_i is throughput of user i
        # Returns 0.0 if input is empty or all throughputs are zero
        if len(throughput_values) == 0:
            return 0.0
        if np.sum(throughput_values) == 0:
            return 0.0
        numerator = np.sum(throughput_values) ** 2
        denominator = len(throughput_values) * np.sum(throughput_values ** 2)
        return numerator / denominator if denominator > 0 else 0.0

    def calculate_total_throughput(self,
                                  throughput_values: np.ndarray) -> float:
        return np.sum(throughput_values)

    def joint_beamforming_power_optimization(self, channel_coeffs, total_power_constraint, beamforming_method, power_optimization_strategy):
        """
        Core joint optimization of beamforming and power allocation.
        Returns the beamforming vectors, power allocation, effective channels, and interference matrix.
        """
        num_users = len(channel_coeffs)
        num_antennas = channel_coeffs.shape[1] if channel_coeffs.ndim > 1 else 1

        # Beamforming design
        if beamforming_method == 'mrt':
            beamforming_vectors = self.beamforming.maximum_ratio_transmission(
                channel_coeffs, total_power_constraint
            )
        elif beamforming_method == 'zf':
            # Convert channel_coeffs from (num_users, num_antennas) to (num_antennas, num_users)
            channel_matrix = np.array(channel_coeffs).T
            beamforming_vectors = self.beamforming.zero_forcing_beamforming(
                channel_matrix, total_power_constraint
            )
            # ZF now returns shape (num_users, num_antennas), which is what we need
        elif beamforming_method == 'random':
            # Create an independent random number generator, not affected by the global seed.
            # This ensures that random beamforming truly exhibits random performance.
            rng = np.random.RandomState(None)  # Use system time as the seed
            
            beamforming_vectors = rng.randn(num_users, num_antennas) + 1j * rng.randn(num_users, num_antennas)
            for i in range(num_users):
                norm = np.linalg.norm(beamforming_vectors[i])
                if norm > 0:
                    beamforming_vectors[i] = beamforming_vectors[i] / norm
            total_power = np.sum(np.linalg.norm(beamforming_vectors, axis=1) ** 2)
            if total_power > 0:
                scale_factor = np.sqrt(total_power_constraint / total_power)
                beamforming_vectors *= scale_factor
            
            # print("Random Debug: Using independent random number generator to avoid fixed seed effects")
        else:
            raise ValueError(f"Unknown beamforming method: {beamforming_method}")

        # Effective channel calculation
        effective_channels = self.calculate_effective_channel(channel_coeffs, beamforming_vectors)

        # Interference matrix calculation
        interference_matrix = self.calculate_interference_matrix(channel_coeffs, beamforming_vectors)

        # Power allocation strategy - FIXED: Use original channel coefficients for proportional allocation
        if power_optimization_strategy == 'equal':
            power_allocation = self.optimize_power_allocation(
                effective_channels, total_power_constraint, method='equal'
            )
        elif power_optimization_strategy == 'proportional':
            # Use original channel coefficients for proportional allocation (not effective channels)
            # Calculate average channel gain for each user from original channel coefficients
            original_channel_gains = []
            for i in range(num_users):
                # Use average channel gain from original multi-antenna channel
                avg_gain = np.mean(np.abs(channel_coeffs[i]) ** 2)
                original_channel_gains.append(avg_gain)
            
            original_channel_gains = np.array(original_channel_gains)
            total_gain = np.sum(original_channel_gains)
            if total_gain > 0:
                power_allocation = original_channel_gains / total_gain
            else:
                power_allocation = np.full(num_users, 1.0 / num_users)
        elif power_optimization_strategy == 'water_filling':
            power_allocation = self.optimize_power_allocation(
                effective_channels, total_power_constraint, method='water_filling'
            )
        else:
            raise ValueError(f"Unknown power optimization strategy: {power_optimization_strategy}")

        # DEBUG: Print power allocation for verification
        # print(f"DEBUG: Strategy={power_optimization_strategy}, Power allocation={power_allocation}")

        return {
            'beamforming_vectors': beamforming_vectors,
            'power_allocation': power_allocation,
            'effective_channels': effective_channels,
            'interference_matrix': interference_matrix
        }

    def calculate_performance_metrics(self, optimization_results, noise_power=1e-10):
        """
        Calculate performance metrics (SINR, throughput, fairness, power efficiency) from optimization results.
        """
        effective_channels = optimization_results['effective_channels']
        interference_matrix = optimization_results['interference_matrix']
        power_allocation = optimization_results['power_allocation']

        # Calculate SINR for all users
        sinr_values = self.calculate_sinr_for_all_users(
            effective_channels,
            interference_matrix,
            power_allocation,
            noise_power
        )
        # Throughput for all users
        throughput_values = self.calculate_throughput_for_all_users(sinr_values)
        # print(f"signal processor throughput_values: {throughput_values}")
        # Fairness index
        fairness_index = self.calculate_fairness_index(throughput_values)
        # Actual total power used
        actual_total_power = np.sum(power_allocation)
        # Power efficiency: total throughput / total power
        power_efficiency = np.sum(throughput_values) / actual_total_power if actual_total_power > 0 else 0.0

        return {
            'sinr_values': sinr_values,
            'spectral_efficiency_values': throughput_values,  
            'total_throughput': np.sum(throughput_values),
            'total_spectral_efficiency': np.sum(throughput_values),  
            'fairness_index': fairness_index,
            'power_efficiency': power_efficiency,
            'actual_total_power': actual_total_power
        }

    def get_joint_optimization_metrics(self, channel_coeffs, total_power_constraint, beamforming_method, power_optimization_strategy, noise_power=1e-10):
        """
        Combines joint beamforming and power optimization with performance metric calculation.
        """
        # print(f"DEBUG: get_joint_optimization_metrics, power_optimization_strategy={power_optimization_strategy}")
        optimization_results = self.joint_beamforming_power_optimization(
            channel_coeffs, total_power_constraint, beamforming_method, power_optimization_strategy
        )
        performance_metrics = self.calculate_performance_metrics(optimization_results, noise_power=noise_power)
        return {**optimization_results, **performance_metrics}

    def calculate_system_throughput(self, uav_position, user_positions, num_antennas, 
                                   total_power_constraint, channel_model, 
                                   beamforming_method='mrt', power_strategy='proportional'):
        """
        Unified interface for calculating system throughput.
        
        This method provides a simplified interface for the environment layer,
        hiding the complexity of algorithm selection and optimization.
        
        Args:
            uav_position: UAV position
            user_positions: Array of user positions
            num_antennas: Number of antennas
            total_power_constraint: Total power constraint
            channel_model: Channel model instance
            beamforming_method: Beamforming method ('mrt', 'zf', 'random')
            power_strategy: Power allocation strategy ('equal', 'proportional', 'water_filling')
            
        Returns:
            Total system throughput
        """
        # Calculate multi-antenna channel coefficients for all users
        channel_coeffs = []
        for user_pos in user_positions:
            channel_coeffs_user = channel_model.calculate_multi_antenna_channel(
                uav_position, user_pos, num_antennas
            )
            channel_coeffs.append(channel_coeffs_user)
        
        channel_coeffs = np.array(channel_coeffs)  # Shape: (num_users, num_antennas)
        
        # print(f"signal processor power_strategy: {power_strategy}")
        # Use joint optimization for best performance
        try:
            performance_metrics = self.get_joint_optimization_metrics(
                channel_coeffs=channel_coeffs,
                total_power_constraint=total_power_constraint,
                beamforming_method=beamforming_method,
                power_optimization_strategy=power_strategy,
                noise_power=channel_model.noise_power
            )
            # print(f"signal processor performance_metrics: {performance_metrics}")
            # Store individual throughputs for user update
            self._last_individual_throughputs = performance_metrics['spectral_efficiency_values']
            
            return performance_metrics['total_spectral_efficiency']
            
        except Exception as e:
            # Fallback to simple MRT if optimization fails
            # print(f"Warning: Joint optimization failed ({e}), using simple MRT fallback")
            print(f"Fallback to MRT: {e}")
            # Simple MRT fallback
            beamforming_vectors = self.beamforming.maximum_ratio_transmission(
                channel_coeffs, total_power_constraint
            )
            
            effective_channels = self.calculate_effective_channel(
                channel_coeffs, beamforming_vectors
            )
            
            sinr_values = self.calculate_sinr(
                effective_channels, beamforming_vectors, channel_model.noise_power
            )
            
            spectral_efficiency_values = self.calculate_throughput_for_all_users(sinr_values)
            
            # Store individual throughputs
            self._last_individual_throughputs = spectral_efficiency_values
            
            return np.sum(spectral_efficiency_values)
    
    def get_last_individual_throughputs(self):
        """Get individual user throughputs from last calculation."""
        return getattr(self, '_last_individual_throughputs', [])

    def __repr__(self) -> str:
        return f"SignalProcessor(antennas={self.num_antennas})"