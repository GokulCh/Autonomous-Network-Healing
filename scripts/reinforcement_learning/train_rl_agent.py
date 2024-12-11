import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random

class SimpleNetworkEnvironment(gym.Env):
    """
    Simplified Gym environment for network healing
    """
    def __init__(self):
        super(SimpleNetworkEnvironment, self).__init__()
        
        # Simplified action space
        self.action_space = spaces.Discrete(3)
        
        # Simplified observation space
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(4,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """
        Reset the network state
        """
        self.network_state = np.random.rand(4)
        self.current_step = 0
        return self.network_state
    
    def step(self, action):
        """
        Perform action and return next state
        """
        # Simple network state modification
        noise = np.random.normal(0, 0.1, self.network_state.shape)
        
        if action == 1:  # Improve network
            self.network_state = np.clip(self.network_state + 0.2 + noise, 0, 1)
        elif action == 2:  # Major reset
            self.network_state = np.clip(np.random.rand(4), 0, 1)
        
        # Calculate reward based on network improvement
        reward = np.mean(self.network_state)
        
        # End episode after 50 steps
        self.current_step += 1
        done = self.current_step >= 50
        
        return self.network_state, reward, done, {}

class SimpleRLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Simplified hyperparameters
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.learning_rate = 0.01
        
        # Simple neural network
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Simplified neural network
        """
        model = Sequential([
            Dense(16, input_dim=self.state_size, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model
    
    def act(self, state):
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def train(self, env, episodes=200):
        """
        Simplified training loop
        """
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Choose and perform action
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # Prepare training data
                target = reward
                target_vec = self.model.predict(state.reshape(1, -1))[0]
                target_vec[action] = target
                
                # Train the model
                self.model.fit(state.reshape(1, -1), target_vec.reshape(1, -1), 
                               epochs=1, verbose=0)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Decay exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= 0.99
            
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
        
        # Save the model
        self.model.save('simple_network_healing_model.h5')

def main():
    # Create environment and agent
    env = SimpleNetworkEnvironment()
    agent = SimpleRLAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )
    
    # Train the agent
    agent.train(env)

if __name__ == '__main__':
    main()