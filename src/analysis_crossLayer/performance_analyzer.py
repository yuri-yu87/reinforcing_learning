"""
Performance Analysis Module

This module handles all performance analysis and monitoring functionalities
that were previously embedded in the environment layer.

Responsibilities:
- Joint optimization results logging and analysis
- Performance metrics calculation and comparison
- Optimization method comparison
- Statistical analysis and reporting
- Visualization and plotting (if needed)

This follows the architectural principle of separating analysis concerns
from core environment simulation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    step: int
    total_throughput: float
    individual_throughputs: List[float]
    power_utilization: float
    fairness_index: float
    power_efficiency: float
    actual_total_power: float
    beamforming_method: str
    power_strategy: str
    timestamp: float


class PerformanceAnalyzer:
    """
    Performance analysis and monitoring for UAV communication systems.
    
    This class handles all performance tracking, analysis, and comparison
    functionality that was previously mixed into the environment layer.
    """
    
    def __init__(self, save_path: Optional[str] = None):
        """
        Initialize performance analyzer.
        
        Args:
            save_path: Optional path to save analysis results
        """
        self.save_path = Path(save_path) if save_path else None
        self.reset()
    
    def reset(self):
        """Reset all tracking data for new episode."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.episode_summary: Dict[str, Any] = {}
        self.method_comparisons: Dict[str, List[PerformanceMetrics]] = {}
    
    def log_step_metrics(self, 
                        step: int,
                        total_throughput: float,
                        individual_throughputs: List[float],
                        power_utilization: float = 1.0,
                        fairness_index: float = 1.0,
                        power_efficiency: float = 0.0,
                        actual_total_power: float = 0.5,
                        beamforming_method: str = 'mrt',
                        power_strategy: str = 'equal',
                        timestamp: float = 0.0):
        """
        Log performance metrics for a single step.
        
        Args:
            step: Current step number
            total_throughput: Total system throughput
            individual_throughputs: List of individual user throughputs
            power_utilization: Power utilization ratio
            fairness_index: Jain's fairness index
            power_efficiency: Power efficiency metric
            actual_total_power: Actual total power used
            beamforming_method: Beamforming method used
            power_strategy: Power allocation strategy
            timestamp: Current timestamp
        """
        metrics = PerformanceMetrics(
            step=step,
            total_throughput=total_throughput,
            individual_throughputs=individual_throughputs.copy(),
            power_utilization=power_utilization,
            fairness_index=fairness_index,
            power_efficiency=power_efficiency,
            actual_total_power=actual_total_power,
            beamforming_method=beamforming_method,
            power_strategy=power_strategy,
            timestamp=timestamp
        )
        
        self.metrics_history.append(metrics)
        
        # Add to method comparison tracking
        method_key = f"{beamforming_method}_{power_strategy}"
        if method_key not in self.method_comparisons:
            self.method_comparisons[method_key] = []
        self.method_comparisons[method_key].append(metrics)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current episode.
        
        Returns:
            Dictionary containing episode summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Extract time series data
        total_throughputs = [m.total_throughput for m in self.metrics_history]
        power_utilizations = [m.power_utilization for m in self.metrics_history]
        fairness_indices = [m.fairness_index for m in self.metrics_history]
        power_efficiencies = [m.power_efficiency for m in self.metrics_history]
        actual_powers = [m.actual_total_power for m in self.metrics_history]
        
        # Calculate summary statistics
        summary = {
            'total_steps': len(self.metrics_history),
            'average_throughput': np.mean(total_throughputs),
            'max_throughput': np.max(total_throughputs),
            'min_throughput': np.min(total_throughputs),
            'std_throughput': np.std(total_throughputs),
            'total_cumulative_throughput': np.sum(total_throughputs),
            
            'average_power_utilization': np.mean(power_utilizations),
            'average_fairness_index': np.mean(fairness_indices),
            'average_power_efficiency': np.mean(power_efficiencies),
            'average_actual_power': np.mean(actual_powers),
            
            'beamforming_method': self.metrics_history[0].beamforming_method,
            'power_strategy': self.metrics_history[0].power_strategy,
            
            # Performance trends
            'throughput_trend': self._calculate_trend(total_throughputs),
            'efficiency_trend': self._calculate_trend(power_efficiencies),
        }
        
        self.episode_summary = summary
        return summary
    
    def compare_methods(self) -> Dict[str, Any]:
        """
        Compare performance across different methods.
        
        Returns:
            Dictionary containing method comparison results
        """
        if not self.method_comparisons:
            return {}
        
        comparison_results = {}
        
        for method_key, metrics_list in self.method_comparisons.items():
            if not metrics_list:
                continue
                
            # Calculate statistics for this method
            throughputs = [m.total_throughput for m in metrics_list]
            power_efficiencies = [m.power_efficiency for m in metrics_list]
            fairness_indices = [m.fairness_index for m in metrics_list]
            
            comparison_results[method_key] = {
                'sample_count': len(metrics_list),
                'avg_throughput': np.mean(throughputs),
                'std_throughput': np.std(throughputs),
                'max_throughput': np.max(throughputs),
                'min_throughput': np.min(throughputs),
                'avg_power_efficiency': np.mean(power_efficiencies),
                'avg_fairness_index': np.mean(fairness_indices),
                'total_throughput': np.sum(throughputs)
            }
        
        # Find best performing method
        if comparison_results:
            best_method = max(comparison_results.items(), 
                            key=lambda x: x[1]['avg_throughput'])
            comparison_results['best_method'] = {
                'name': best_method[0],
                'avg_throughput': best_method[1]['avg_throughput']
            }
        
        return comparison_results
    
    def get_optimization_analysis(self) -> Dict[str, Any]:
        """
        Analyze optimization effectiveness.
        
        Returns:
            Dictionary containing optimization analysis
        """
        if not self.metrics_history:
            return {}
        
        # Analyze power utilization efficiency
        power_utils = [m.power_utilization for m in self.metrics_history]
        power_efficiencies = [m.power_efficiency for m in self.metrics_history]
        
        # Analyze fairness
        fairness_values = [m.fairness_index for m in self.metrics_history]
        
        # Analyze convergence (if applicable)
        throughputs = [m.total_throughput for m in self.metrics_history]
        convergence_analysis = self._analyze_convergence(throughputs)
        
        return {
            'power_optimization': {
                'avg_utilization': np.mean(power_utils),
                'utilization_stability': 1.0 - np.std(power_utils),  # Higher is better
                'avg_efficiency': np.mean(power_efficiencies),
                'efficiency_trend': self._calculate_trend(power_efficiencies)
            },
            'fairness_analysis': {
                'avg_fairness': np.mean(fairness_values),
                'fairness_stability': 1.0 - np.std(fairness_values),
                'fairness_trend': self._calculate_trend(fairness_values)
            },
            'convergence_analysis': convergence_analysis,
            'optimization_score': self._calculate_optimization_score()
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend analysis
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_convergence(self, values: List[float]) -> Dict[str, Any]:
        """Analyze convergence properties of a metric series."""
        if len(values) < 10:
            return {'status': 'insufficient_data'}
        
        # Calculate rolling standard deviation to detect convergence
        window_size = min(10, len(values) // 4)
        rolling_stds = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            rolling_stds.append(np.std(window))
        
        if not rolling_stds:
            return {'status': 'insufficient_data'}
        
        # Check if variance is decreasing (convergence)
        trend = self._calculate_trend(rolling_stds)
        final_variance = rolling_stds[-1]
        
        return {
            'status': 'converging' if trend == 'decreasing' else 'not_converging',
            'final_variance': final_variance,
            'variance_trend': trend,
            'convergence_score': max(0, 1.0 - final_variance)  # Higher is better
        }
    
    def _calculate_optimization_score(self) -> float:
        """Calculate overall optimization effectiveness score."""
        if not self.metrics_history:
            return 0.0
        
        # Weighted combination of different metrics
        throughputs = [m.total_throughput for m in self.metrics_history]
        power_efficiencies = [m.power_efficiency for m in self.metrics_history]
        fairness_indices = [m.fairness_index for m in self.metrics_history]
        
        # Normalize metrics to [0, 1] range
        throughput_score = np.mean(throughputs) / max(np.max(throughputs), 1e-6)
        efficiency_score = np.mean(power_efficiencies) / max(np.max(power_efficiencies), 1e-6)
        fairness_score = np.mean(fairness_indices)
        
        # Weighted average (can be tuned)
        optimization_score = (0.5 * throughput_score + 
                            0.3 * efficiency_score + 
                            0.2 * fairness_score)
        
        return float(np.clip(optimization_score, 0.0, 1.0))
    
    def save_results(self, filename: Optional[str] = None):
        """Save analysis results to file."""
        if not self.save_path:
            return
        
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'episode_summary': self.get_episode_summary(),
            'method_comparison': self.compare_methods(),
            'optimization_analysis': self.get_optimization_analysis(),
            'raw_metrics': [asdict(m) for m in self.metrics_history]
        }
        
        # Save to JSON file
        filename = filename or f"performance_analysis_episode_{len(self.metrics_history)}.json"
        save_file = self.save_path / filename
        
        with open(save_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Performance analysis saved to {save_file}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load analysis results from file."""
        if not self.save_path:
            raise ValueError("Save path not configured")
        
        load_file = self.save_path / filename
        
        with open(load_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def print_summary(self):
        """Print a human-readable summary of performance analysis."""
        summary = self.get_episode_summary()
        comparison = self.compare_methods()
        optimization = self.get_optimization_analysis()
        
        print("=" * 60)
        print("📊 PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        if summary:
            print(f"\n📈 Episode Statistics:")
            print(f"  Total Steps: {summary['total_steps']}")
            print(f"  Average Throughput: {summary['average_throughput']:.4f}")
            print(f"  Peak Throughput: {summary['max_throughput']:.4f}")
            print(f"  Total Cumulative: {summary['total_cumulative_throughput']:.2f}")
            print(f"  Method: {summary['beamforming_method']} + {summary['power_strategy']}")
        
        if comparison:
            print(f"\n🔄 Method Comparison:")
            for method, stats in comparison.items():
                if method != 'best_method':
                    print(f"  {method}: Avg={stats['avg_throughput']:.4f}, "
                          f"Samples={stats['sample_count']}")
            
            if 'best_method' in comparison:
                best = comparison['best_method']
                print(f"  🏆 Best Method: {best['name']} ({best['avg_throughput']:.4f})")
        
        if optimization:
            print(f"\n⚡ Optimization Analysis:")
            if 'power_optimization' in optimization:
                po = optimization['power_optimization']
                print(f"  Power Utilization: {po['avg_utilization']:.3f}")
                print(f"  Power Efficiency: {po['avg_efficiency']:.4f}")
            
            if 'fairness_analysis' in optimization:
                fa = optimization['fairness_analysis']
                print(f"  Fairness Index: {fa['avg_fairness']:.3f}")
            
            print(f"  Overall Score: {optimization.get('optimization_score', 0):.3f}")
        
        print("=" * 60)
