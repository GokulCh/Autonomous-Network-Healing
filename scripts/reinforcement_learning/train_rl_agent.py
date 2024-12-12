import subprocess
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any

def load_model_with_custom_objects(model_path: str):
    """
    Safely load a Keras model with custom loss functions and error handling
    
    Parameters:
    - model_path: Path to the saved Keras model
    
    Returns:
    - Loaded Keras model or None if loading fails
    """
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        # Add any other custom loss functions or layers here
    }
    
    try:
        print(f"Attempting to load model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects
        )
        print("Model loaded successfully!")
        return model
    except ValueError as e:
        print(f"ValueError: {e}")
        print("There may be an issue with the model file or custom objects.")
        return None
    except TypeError as e:
        print(f"TypeError: {e}")
        print("Possible issue with the custom objects or model compatibility.")
        return None
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

class NS3NetworkSimulation:
    def __init__(self, 
                 autoencoder_model_path: str, 
                 rl_model_path: str):
        """
        Initialize NS3 simulation with pre-trained ML models
        
        Parameters:
        - autoencoder_model_path: Path to saved autoencoder model
        - rl_model_path: Path to saved reinforcement learning model
        """
        # Load pre-trained models with custom loading function
        self.autoencoder = load_model_with_custom_objects(autoencoder_model_path)
        self.rl_model = load_model_with_custom_objects(rl_model_path)
        
        # Validate model loading
        if self.autoencoder is None or self.rl_model is None:
            raise ValueError("Failed to load one or more ML models. Check model paths and compatibility.")
        
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
        
        Parameters:
        - scenario: Type of network scenario to simulate
        
        Returns:
        - Dictionary of simulation parameters
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
        
        Parameters:
        - scenario: Network scenario to simulate
        
        Returns:
        - Simulation results and metrics
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
        
        Parameters:
        - output: Raw simulation output string
        
        Returns:
        - Dictionary of parsed metrics
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
        
        Parameters:
        - simulation_data: Network traffic data from NS3
        
        Returns:
        - List of boolean anomaly flags
        """
        try:
            # Reconstruct input data
            reconstructed = self.autoencoder.predict(simulation_data)
            
            # Calculate reconstruction error
            mse = np.mean(np.square(simulation_data - reconstructed), axis=1)
            
            # Set anomaly threshold (95th percentile)
            threshold = np.percentile(mse, 95)
            
            return mse > threshold
        except Exception as e:
            print(f"Anomaly detection error: {e}")
            return []
    
    def apply_corrective_actions(self, anomalies: List[bool], metrics: Dict[str, Any]):
        """
        Use RL model to determine and apply corrective actions
        
        Parameters:
        - anomalies: List of detected anomalies
        - metrics: Network performance metrics
        """
        try:
            # Convert metrics and anomalies to state representation
            state = self._prepare_rl_state(anomalies, metrics)
            
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
    
    def _prepare_rl_state(self, anomalies: List[bool], metrics: Dict[str, Any]) -> np.ndarray:
        """
        Prepare state representation for RL model
        
        Parameters:
        - anomalies: List of detected anomalies
        - metrics: Network performance metrics
        
        Returns:
        - Normalized state vector
        """
        try:
            state_features = [
                metrics.get('packet_loss_rate', 0),
                metrics.get('average_latency', 0),
                metrics.get('network_throughput', 0),
                sum(anomalies) / len(anomalies) if anomalies else 0  # Proportion of anomalies
            ]
            
            return np.array(state_features)
        except Exception as e:
            print(f"Error preparing RL state: {e}")
            return np.zeros(4)
    
    def comprehensive_network_test(self):
        """
        Run comprehensive network testing across all scenarios
        
        Returns:
        - Dictionary of test results for each scenario
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
                
                # Apply corrective actions
                self.apply_corrective_actions(anomalies, simulation_metrics)
                
                test_results[scenario] = {
                    'metrics': simulation_metrics,
                    'anomalies_detected': sum(anomalies)
                }
        
        return test_results

def main():
    try:
        # Initialize simulation framework
        network_tester = NS3NetworkSimulation(
            autoencoder_model_path='network_anomaly_autoencoder.h5',
            rl_model_path='network_healing_rl_model.h5'
        )
        
        # Run comprehensive network tests
        test_results = network_tester.comprehensive_network_test()
        
        # Output results
        for scenario, results in test_results.items():
            print(f"\nScenario: {scenario}")
            print(f"Performance Metrics: {results['metrics']}")
            print(f"Anomalies Detected: {results['anomalies_detected']}")
    
    except Exception
