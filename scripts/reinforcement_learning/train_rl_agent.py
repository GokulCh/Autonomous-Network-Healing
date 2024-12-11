import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

class NetworkEnvironment(gym.Env):
    """
    Custom Gym environment simulating a network healing scenario
    """
    def __init__(self, num_nodes=10):
        super(NetworkEnvironment, self).__init__()
        
        # Action space: 
        # 0: Do nothing
        # 1: Reroute traffic
        # 2: Isolate problematic node
        # 3: Adjust bandwidth
        # 4: Reset network connection
        self.action_space = spaces.Discrete(5)
        
        # Observation space: network state features
        # Includes: node health, traffic load, packet loss, latency
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(num_nodes * 4,), 
            dtype=np.float32
        )
        
        # Initialize network state
        self.num_nodes = num_nodes
        self.reset()
    
    def reset(self):
        """
        Reset the network to initial state
        """
        # Randomly initialize network state
        self.network_state = np.random.rand(self.num_nodes * 4)
        self.current_step = 0
        self.total_network_health = np.mean(self.network_state[::4])
        return self.network_state
    
    def _simulate_network_response(self, action):
        """
        Simulate how the network responds to different actions
        """
        # Modify network state based on action
        noise = np.random.normal(0, 0.1, self.network_state.shape)
        
        if action == 1:  # Reroute traffic
            # Redistribute traffic load
            self.network_state[1::4] = np.clip(
                self.network_state[1::4] * 0.7 + noise[1::4], 
                0, 1
            )
        elif action == 2:  # Isolate problematic node
            # Find most problematic node
            worst_node_index = np.argmin(self.network_state[::4]) * 4
            self.network_state[worst_node_index] = 0  # Mark as isolated
        elif action == 3:  # Adjust bandwidth
            # Increase bandwidth for low-performing nodes
            self.network_state[2::4] = np.clip(
                self.network_state[2::4] * 1.2 + noise[2::4], 
                0, 1
            )
        elif action == 4:  # Reset network connection
            # Partial reset of network state
            self.network_state = np.clip(
                self.network_state * 0.5 + np.random.rand(self.num_nodes * 4) * 0.5, 
                0, 1
            )
        
        return self.network_state
    
    def step(self, action):
        """
        Execute an action and return next state, reward, done, info
        """
        # Simulate network response to action
        next_state = self._simulate_network_response(action)
        
        # Calculate network health
        new_network_health = np.mean(next_state[::4])
        
        # Reward function
        reward = new_network_health - self.total_network_health
        
        # Determine if episode is done
        self.current_step += 1
        done = self.current_step >= 100  # Max episode length
        
        # Update total network health
        self.total_network_health = new_network_health
        
        return next_state, reward, done, {}

class DeepQLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience replay
        self.memory = []
        self.batch_size = 32
        self.memory_size = 2000
        
        # Hyperparameters
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Build neural network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Neural network for Q-learning
        """
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """
        Copy weights from main model to target model
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Experience replay for learning
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare training data
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Predict Q-values
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = (
                    rewards[i] + self.gamma * np.max(next_q_values[i])
                )
        
        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=1000):
        """
        Train the RL agent
        """
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Select and perform action
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Perform learning
                self.replay()
            
            # Periodically update target model
            if episode % 10 == 0:
                self.update_target_model()
            
            # Print progress
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
        
        # Save the trained model
        self.model.save('network_healing_rl_model.h5')

def main():
    # Create network environment
    env = NetworkEnvironment(num_nodes=10)
    
    # Initialize RL agent
    agent = DeepQLearningAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    # Train the agent
    agent.train(env)

if __name__ == '__main__':
    main()