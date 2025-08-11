import numpy as np
from typing import Tuple, Optional, List


class ChannelModel:
    """
    Channel model for UAV-ground user communication.
    
    Implements Line-of-Sight (LoS) path loss model following the mathematical formulas:
    - hk = sqrt(L0/ d_k^η) * h_k^LoS 
    - L0 = (λ/(4π))^2
    - h_k^LoS = e^(j * 2π * d_k / λ)
    """
    
    def __init__(self, 
                 frequency: float = 2.4e9,  # 2.4 GHz
                 path_loss_exponent: float = 2.5,
                 noise_power: float = -100.0,  # dB
                 seed: Optional[int] = None):
        """
        Initialize channel model.
        
        Args:
            frequency: Carrier frequency in Hz
            path_loss_exponent: Path loss exponent η
            noise_power: Noise power in dB
            seed: Random seed for reproducibility
        """
        self.frequency = frequency
        self.path_loss_exponent = path_loss_exponent
        self.noise_power_db = noise_power
        self.noise_power = 10**(noise_power / 10)  # Convert to linear scale
        # Speed of light
        self.c = 3e8
        self.wavelength = self.c / self.frequency
        self.antenna_spacing = self.wavelength/2  # Half wavelength spacing
        
        # Reference path loss term L0 = (λ/(4π))^2 (Equation 2)
        self.L0 = (self.wavelength / (4 * np.pi)) ** 2
        
        if seed is not None:
            np.random.seed(seed)

    def calculate_distance(self, uav_position: np.ndarray, user_position: np.ndarray) -> float:
        """
        Calculate 3D distance between UAV and user.
        
        Args:
            uav_position: UAV 3D position
            user_position: User 3D position
            
        Returns:
            float: Distance in meters
        """
        return np.linalg.norm(uav_position - user_position)
    
    def calculate_channel_coefficient(self, 
                                    uav_position: np.ndarray, 
                                    user_position: np.ndarray,
                                    antenna_index: int = 0,
                                    num_antennas: int = 1) -> complex:
        """
        Calculate channel coefficient for single antenna or specific antenna in ULA array.
        
        This function is primarily used for single antenna scenarios (num_antennas=1),
        but can also handle multi-antenna cases by calculating the actual antenna position.
        
        Args:
            uav_position: UAV 3D position (center of antenna array for multi-antenna)
            user_position: User 3D position
            antenna_index: Index of the antenna element (0 to num_antennas-1)
            num_antennas: Total number of antennas in the array (default: 1)
            
        Returns:
            complex: Channel coefficient for the specified antenna
        """
        # For single antenna case (most common usage), use simple distance calculation
        if num_antennas == 1:
            distance = self.calculate_distance(uav_position, user_position)
        else:
            # For multi-antenna case, calculate actual antenna position in ULA
            antenna_y_offset = (antenna_index - (num_antennas - 1) / 2) * self.antenna_spacing
            antenna_position = uav_position.copy()
            antenna_position[1] += antenna_y_offset  # Y-axis offset
            distance = self.calculate_distance(antenna_position, user_position)
        
        if distance <= 0:
            return 0.0
        
        # Calculate LoS component: h_k^LoS = e^(j * 2π * d_k / λ)
        los_phase = 2 * np.pi * distance / self.wavelength
        los_component = np.exp(1j * los_phase)
        
        # Channel coefficient: hk = sqrt(L0/ d_k^η) * h_k^LoS
        path_loss_factor = np.sqrt(self.L0 / (distance ** self.path_loss_exponent))
        channel_coeff = path_loss_factor * los_component
        
        return channel_coeff

    # for figure 1 and 2
    def calculate_channel_coefficient_distance(self, 
                                    distance: float) -> complex:
        
        if distance <= 0:
            return 0.0
        
        # Calculate LoS component: h_k^LoS = e^(j * 2π * d_k / λ)
        los_phase = 2 * np.pi * distance / self.wavelength
        los_component = np.exp(1j * los_phase)
        
        # Channel coefficient: hk = sqrt(L0/ d_k^η) * h_k^LoS
        path_loss_factor = np.sqrt(self.L0 / (distance ** self.path_loss_exponent))
        channel_coeff = path_loss_factor * los_component
        
        return channel_coeff
    
    def calculate_multi_antenna_channel(self,
                                      uav_position: np.ndarray,
                                      user_position: np.ndarray,
                                      num_antennas: int) -> np.ndarray:
        """
        Calculate channel coefficients for all antennas based on ULA geometry.
        
        According to point 4 in the document: UAV is equipped with Nt antennas
        organized in a uniform linear array along the Y-axis.
        
        Args:
            uav_position: UAV 3D position (center of antenna array)
            user_position: User 3D position
            num_antennas: Number of antennas (Nt)
            
        Returns:
            np.ndarray: Channel coefficients for all antennas
        """
        channel_coeffs = np.zeros(num_antennas, dtype=complex)
        
        # Calculate antenna positions along Y-axis
        # Use the class parameter for antenna spacing
        antenna_spacing = self.antenna_spacing  # λ/2 spacing
        total_array_length = (num_antennas - 1) * antenna_spacing
        
        # Calculate antenna positions relative to UAV center
        for i in range(num_antennas):
            # Antenna position: y = (i - (Nt-1)/2) * λ/2
            # This centers the array at UAV position
            antenna_y_offset = (i - (num_antennas - 1) / 2) * antenna_spacing
            
            # Calculate actual antenna position
            antenna_position = uav_position.copy()
            antenna_position[1] += antenna_y_offset  # Y-axis offset
            
            # Calculate channel coefficient for this specific antenna
            channel_coeffs[i] = self.calculate_channel_coefficient_for_antenna(
                antenna_position, user_position
            )
        
        return channel_coeffs
    
    def calculate_channel_coefficient_for_antenna(self,
                                                antenna_position: np.ndarray,
                                                user_position: np.ndarray) -> complex:
        """
        Calculate channel coefficient for a specific antenna position.
        
        This method calculates the channel coefficient based on the actual
        distance between the antenna and the user, following the path loss model.
        
        Args:
            antenna_position: 3D position of the specific antenna
            user_position: User 3D position
            
        Returns:
            complex: Channel coefficient for this antenna
        """
        # Calculate distance between this antenna and user
        distance = self.calculate_distance(antenna_position, user_position)
        
        if distance <= 0:
            return 0.0
        
        # Calculate LoS component: h_k^LoS = e^(j * 2π * d_k / λ)
        los_phase = 2 * np.pi * distance / self.wavelength
        los_component = np.exp(1j * los_phase)
        
        # Channel coefficient: hk = sqrt(L0/ d_k^η) * h_k^LoS
        path_loss_factor = np.sqrt(self.L0 / (distance ** self.path_loss_exponent))
        channel_coeff = path_loss_factor * los_component
        
        return channel_coeff
    
    def calculate_snr(self, 
                     channel_coeff: complex, 
                     transmit_power: float) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) following Equation 3.
        
        Args:
            channel_coeff: Channel coefficient
            transmit_power: Transmit power in Watts
            
        Returns:
            float: SNR in linear scale
        """
        # SNR = E[|hk*xk(t)|^2] / E[|n(t)|^2] = P * |hk|^2 / σ^2
        signal_power = transmit_power * (np.abs(channel_coeff) ** 2)
        snr = signal_power / self.noise_power
        
        return snr
    
    def calculate_sinr(self,
                      desired_channel: complex,
                      interference_channels: List[complex],
                      desired_power: float,
                      interference_powers: List[float]) -> float:
        """
        Calculate Signal-to-Interference-plus-Noise Ratio (SINR).
        
        Args:
            desired_channel: Channel coefficient for desired signal
            interference_channels: List of channel coefficients for interfering signals
            desired_power: Transmit power for desired signal
            interference_powers: List of transmit powers for interfering signals
            
        Returns:
            float: SINR in linear scale
        """
        # Desired signal power
        desired_signal_power = desired_power * (np.abs(desired_channel) ** 2)
        
        # Interference power
        interference_power = 0.0
        for i, (channel, power) in enumerate(zip(interference_channels, interference_powers)):
            interference_power += power * (np.abs(channel) ** 2)
        
        # SINR = desired_signal_power / (interference_power + noise_power)
        sinr = desired_signal_power / (interference_power + self.noise_power)
        
        return sinr
    
    def calculate_throughput(self, sinr: float) -> float:
        """
        Calculate spectral efficiency using Shannon capacity formula.
        
        Args:
            sinr: Signal-to-Interference-plus-Noise Ratio in linear scale
            
        Returns:
            float: Spectral efficiency in bits per channel use
        """
        # Shannon capacity: R_k(t) = log2(1 + SINR_k(t)) [bits per channel use]
        if sinr <= 0:
            return 0.0
        
        spectral_efficiency = np.log2(1 + sinr)
        return spectral_efficiency
    
    def get_channel_info(self, 
                        uav_position: np.ndarray, 
                        user_positions: np.ndarray,
                        transmit_powers: np.ndarray,
                        num_antennas: int = 1) -> dict:
        """
        Get comprehensive channel information for unified single/multi-user scenarios.
        
        This unified interface handles both single-user (SNR) and multi-user (SINR) scenarios.
        For single user: user_positions.shape = (1, 3)
        For multi users: user_positions.shape = (N, 3)
        
        Args:
            uav_position: UAV 3D position
            user_positions: User 3D position(s) - can be single user or multiple users
            transmit_powers: Transmit power(s) - can be single value or array
            num_antennas: Number of antennas
            
        Returns:
            dict: Channel information including SNR/SINR, throughput for each user
        """
        # Normalize inputs to handle both single and multi-user cases
        if user_positions.ndim == 1:
            # Single user case: (3,) -> (1, 3)
            user_positions = user_positions.reshape(1, 3)
            transmit_powers = np.array([transmit_powers]) if np.isscalar(transmit_powers) else transmit_powers.reshape(1)
            is_single_user = True
        else:
            # Multi-user case: (N, 3)
            is_single_user = False
        
        num_users = len(user_positions)
        
        # Calculate channel coefficients for all users
        all_channel_coeffs = []
        distances = []
        
        for user_pos in user_positions:
            distance = self.calculate_distance(uav_position, user_pos)
            distances.append(distance)
            
            if num_antennas == 1:
                channel_coeff = self.calculate_channel_coefficient(uav_position, user_pos)
                all_channel_coeffs.append(channel_coeff)
            else:
                channel_coeffs = self.calculate_multi_antenna_channel(
                    uav_position, user_pos, num_antennas
                )
                # Use average channel coefficient for multi-antenna
                avg_coeff = np.mean(channel_coeffs)
                all_channel_coeffs.append(avg_coeff)
        
        # Calculate SNR/SINR for each user
        signal_quality_values = []
        throughput_values = []
        
        for i in range(num_users):
            desired_channel = all_channel_coeffs[i]
            desired_power = transmit_powers[i]
            
            if is_single_user or num_users == 1:
                # Single user case: use SNR (no interference)
                signal_quality = self.calculate_snr(desired_channel, desired_power)
            else:
                # Multi-user case: use SINR (with interference)
                interference_channels = []
                interference_powers = []
                
                for j in range(num_users):
                    if j != i:  # Other users are interference
                        interference_channels.append(all_channel_coeffs[j])
                        interference_powers.append(transmit_powers[j])
                
                signal_quality = self.calculate_sinr(
                    desired_channel, interference_channels,
                    desired_power, interference_powers
                )
            
            signal_quality_values.append(signal_quality)
            throughput_values.append(self.calculate_throughput(signal_quality))
        
        # Prepare return dictionary
        result = {
            'distances': distances,
            'channel_coefficients': all_channel_coeffs,
            'throughput_values': throughput_values,
            'total_throughput': sum(throughput_values),
            'num_users': num_users,
            'is_single_user': is_single_user
        }
        
        # Add appropriate signal quality metrics
        if is_single_user or num_users == 1:
            # Single user: return SNR metrics
            result.update({
                'snr': signal_quality_values[0],
                'snr_db': 10 * np.log10(signal_quality_values[0]) if signal_quality_values[0] > 0 else -float('inf'),
                'throughput': throughput_values[0]
            })
        else:
            # Multi-user: return SINR metrics
            result.update({
                'sinr_values': signal_quality_values,
                'sinr_db_values': [10 * np.log10(sinr) if sinr > 0 else -float('inf') 
                                  for sinr in signal_quality_values],
                'avg_sinr': np.mean(signal_quality_values),
                'min_sinr': min(signal_quality_values),
                'max_sinr': max(signal_quality_values)
            })
        
        return result
    
    def get_multi_user_channel_info(self,
                                   uav_position: np.ndarray,
                                   user_positions: List[np.ndarray],
                                   transmit_powers: List[float],
                                   num_antennas: int = 1) -> dict:
        """
        Get comprehensive channel information for multi-user scenario with SINR calculation.
        
        This is the main interface for multi-user scenarios where interference matters.
        For backward compatibility - now delegates to unified get_channel_info().
        
        Args:
            uav_position: UAV 3D position
            user_positions: List of user 3D positions
            transmit_powers: List of transmit powers for each user
            num_antennas: Number of antennas
            
        Returns:
            dict: Channel information including SINR, throughput for each user
        """
        # Convert to numpy arrays for unified interface
        user_positions_array = np.array(user_positions)
        transmit_powers_array = np.array(transmit_powers)
        
        return self.get_channel_info(uav_position, user_positions_array, transmit_powers_array, num_antennas)
    
    def get_beamforming_channel_info(self,
                                   uav_position: np.ndarray,
                                   user_positions: List[np.ndarray],
                                   transmit_power: float,
                                   num_antennas: int) -> dict:
        """
        Get channel information optimized for beamforming scenarios.
        
        This interface is specifically designed for beamforming applications
        where we need channel coefficients for all antennas and all users.
        
        Args:
            uav_position: UAV 3D position
            user_positions: List of user 3D positions
            transmit_power: Total transmit power (will be allocated by beamforming)
            num_antennas: Number of antennas
            
        Returns:
            dict: Channel information optimized for beamforming
        """
        num_users = len(user_positions)
        
        # Calculate multi-antenna channel coefficients for all users
        channel_matrix = np.zeros((num_users, num_antennas), dtype=complex)
        distances = []
        
        for i, user_pos in enumerate(user_positions):
            distance = self.calculate_distance(uav_position, user_pos)
            distances.append(distance)
            
            channel_coeffs = self.calculate_multi_antenna_channel(
                uav_position, user_pos, num_antennas
            )
            channel_matrix[i, :] = channel_coeffs
        
        # Calculate individual SNR values (without interference)
        snr_values = []
        for i in range(num_users):
            # Use average channel coefficient for SNR calculation
            avg_coeff = np.mean(channel_matrix[i, :])
            snr = self.calculate_snr(avg_coeff, transmit_power / num_users)
            snr_values.append(snr)
        
        return {
            'channel_matrix': channel_matrix,  # Shape: (num_users, num_antennas)
            'distances': distances,
            'snr_values': snr_values,
            'snr_db_values': [10 * np.log10(snr) if snr > 0 else -float('inf') 
                             for snr in snr_values],
            'num_users': num_users,
            'num_antennas': num_antennas,
            'total_power': transmit_power
        }
    
    def __repr__(self) -> str:
        return f"ChannelModel(f={self.frequency/1e9:.1f}GHz, η={self.path_loss_exponent}, σ²={self.noise_power_db}dB, λ={self.wavelength:.3f}m)" 