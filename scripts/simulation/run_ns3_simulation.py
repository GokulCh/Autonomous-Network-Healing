import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from typing import List, Dict, Any

# Load pre-trained models
autoencoder = load_model('models/autoencoder/network_anomaly_autoencoder', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(pd.read_csv('datasets/processed_data/processed_train_features.csv').values)

class NS3NetworkSimulation:
    def __init__(self, rl_model_path: str):
        """
        Initialize NS3 simulation with pre-trained RL model
        """
        self.rl_model = load_model(rl_model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        
        # Simulation configurations
        self.simulation_scenarios = [
            'traffic_congestion',
            'link_failure', 
            'node_isolation', 
            'bandwidth_limitation',
            'packet_loss'
        ]
    
    def generate_ns3_config(self, scenario: str) -> Dict[str, Any]:
        """
        Generate NS3 simulation configuration for different network scenarios
        """
        scenarios_config = {
            'traffic_congestion': {
                'nodes': 50,
                'traffic_rate': 0.8,  # High traffic load
                'congestion_points': 3,
                'duration': 300  # seconds
            },
            'link_failure': {
                'nodes': 30,
                'failure_probability': 0.2,
                'recovery_time': 60,  # seconds
                'duration': 300
            },
            'node_isolation': {
                'nodes': 40,
                'isolation_nodes': 2,
                'reconnection_strategy': 'dynamic',
                'duration': 300
            },
            'bandwidth_limitation': {
                'nodes': 25,
                'bandwidth_limit': 0.5,  # 50% of normal capacity
                'duration': 300
            },
            'packet_loss': {
                'nodes': 35,
                'loss_rate': 0.15,  # 15% packet loss
                'duration': 300
            }
        }
        
        return scenarios_config.get(scenario, {})
    
    def run_ns3_simulation(self, scenario: str) -> Dict[str, Any]:
        """
        Run NS3 simulation for a specific network scenario
        """
        # Generate scenario configuration
        config = self.generate_ns3_config(scenario)
        
        # Create NS3 simulation command
        ns3_command = [
            './ns3-simulator',  # Assumes compiled NS3 executable
            f'--scenario={scenario}',
            f'--nodes={config["nodes"]}',
            f'--duration={config["duration"]}'
        ]
        
        # Additional scenario-specific parameters
        for key, value in config.items():
            if key not in ['nodes', 'duration']:
                ns3_command.append(f'--{key}={value}')
        
        try:
            # Run NS3 simulation
            result = subprocess.run(
                ns3_command, 
                capture_output=True, 
                text=True, 
                timeout=600  # 10-minute timeout
            )
            
            # Parse simulation output
            simulation_metrics = self._parse_simulation_output(result.stdout)
            
            return simulation_metrics
        
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed: {e}")
            print(f"Error output: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            print("Simulation timed out")
            return None
    
    def _parse_simulation_output(self, output: str) -> Dict[str, Any]:
        """
        Parse NS3 simulation output into structured metrics
        """
        try:
            metrics = {
                'total_packets': int(output.split('total_packets:')[1].split('\n')[0]),
                'packet_loss_rate': float(output.split('packet_loss_rate:')[1].split('\n')[0]),
                'average_latency': float(output.split('average_latency:')[1].split('\n')[0]),
                'network_throughput': float(output.split('network_throughput:')[1].split('\n')[0])
            }
            return metrics
        except (IndexError, ValueError) as e:
            print(f"Error parsing simulation output: {e}")
            print(f"Problematic output: {output}")
            return None
    
    def detect_anomalies(self, simulation_data: np.ndarray) -> List[bool]:
        """
        Use autoencoder to detect anomalies in simulation data
        """
        try:
            # Reconstruct input data
            reconstructed = autoencoder.predict(simulation_data)
            
            # Calculate reconstruction error
            mse = np.mean(np.square(simulation_data - reconstructed), axis=1)
            
            # Set anomaly threshold (95th percentile)
            threshold = np.percentile(mse, 95)
            
            return mse > threshold
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return []
    
    def cluster_data(self, simulation_data: np.ndarray) -> np.ndarray:
        """
        Use clustering model to cluster the simulation data
        """
        return kmeans.predict(simulation_data)
    
    def apply_corrective_actions(self, anomalies: List[bool], metrics: Dict[str, Any], clusters: np.ndarray):
        """
        Use RL model to determine and apply corrective actions
        """
        try:
            # Convert metrics, anomalies, and clusters to state representation
            state = self._prepare_rl_state(anomalies, metrics, clusters)
            
            # Predict best action using RL model
            action = np.argmax(self.rl_model.predict(state.reshape(1, -1), verbose=0))
            
            # Map action to corrective strategy
            corrective_actions = {
                0: "do_nothing",
                1: "reroute_traffic",
                2: "isolate_nodes",
                3: "adjust_bandwidth",
                4: "reset_connections"
            }
            
            print(f"Recommended Action: {corrective_actions[action]}")
        except Exception as e:
            print(f"Error applying corrective actions: {e}")
    
    def _prepare_rl_state(self, anomalies: List[bool], metrics: Dict[str, Any], clusters: np.ndarray) -> np.ndarray:
        """
        Prepare state representation for RL model
        """
        try:
            state_features = [
                metrics.get('packet_loss_rate', 0),
                metrics.get('average_latency', 0),
                metrics.get('network_throughput', 0),
                sum(anomalies) / len(anomalies) if anomalies else 0,  # Proportion of anomalies
                np.mean(clusters)  # Mean cluster label
            ]
            
            return np.array(state_features)
        except Exception as e:
            print(f"Error preparing RL state: {e}")
            return np.zeros(5)
    
    def comprehensive_network_test(self):
        """
        Run comprehensive network testing across all scenarios
        """
        test_results = {}
        
        for scenario in self.simulation_scenarios:
            print(f"Testing Scenario: {scenario}")
            
            # Run NS3 simulation
            simulation_metrics = self.run_ns3_simulation(scenario)
            
            if simulation_metrics:
                # Load simulated network data 
                # (in real scenario, this would come from NS3 output)
                simulation_data = np.random.rand(100, 10)  # Placeholder
                
                # Detect anomalies
                anomalies = self.detect_anomalies(simulation_data)
                
                # Cluster data
                clusters = self.cluster_data(simulation_data)
                
                # Apply corrective actions
                self.apply_corrective_actions(anomalies, simulation_metrics, clusters)
                
                test_results[scenario] = {
                    'metrics': simulation_metrics,
                    'anomalies_detected': sum(anomalies)
                }
        
        return test_results

def main():
    try:
        # Initialize simulation framework
        network_tester = NS3NetworkSimulation(
            rl_model_path='models/reinforcement_learning/network_healing_rl_model.h5'
        )
        
        # Run comprehensive network tests
        test_results = network_tester.comprehensive_network_test()
        
        # Output results
        for scenario, results in test_results.items():
            print(f"\nScenario: {scenario}")
            print(f"Performance Metrics: {results['metrics']}")
            print(f"Anomalies Detected: {results['anomalies_detected']}")
    
    except Exception as e:
        print(f"Simulation framework initialization failed: {e}")

if __name__ == '__main__':
    main()