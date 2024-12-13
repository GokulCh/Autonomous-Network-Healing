import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import random

# Registering custom serialization for 'mse' to prevent deserialization issues
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

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
            low=0.0, high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Reset the network state
        """
        self.network_state = np.random.rand(4).astype(np.float32)
        self.current_step = 0
        return self.network_state

    def step(self, action):
        """
        Perform action and return next state
        """
        noise = np.random.normal(0, 0.1, self.network_state.shape).astype(np.float32)

        if action == 1:  # Improve network
            self.network_state = np.clip(self.network_state + 0.2 + noise, 0, 1)
        elif action == 2:  # Major reset
            self.network_state = np.random.rand(4).astype(np.float32)

        # Calculate reward based on network improvement
        reward = np.mean(self.network_state)

        # End episode after 50 steps
        self.current_step += 1
        done = self.current_step >= 50

        return self.network_state, reward, done, {}

    def render(self, mode='human'):
        """
        Render the current state of the environment
        """
        print(f"Step: {self.current_step}, State: {self.network_state}")

class SimpleRLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Simplified hyperparameters
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Simple neural network
        self.model = self._build_model()

    def _build_model(self):
        """
        Simplified neural network
        """
        model = Sequential([
            Dense(32, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=mse, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        """
        Epsilon-greedy action selection
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
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
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                # Update Q-value using Bellman equation
                target = reward
                if not done:
                    next_q_values = self.model.predict(next_state.reshape(1, -1), verbose=0)
                    target += self.gamma * np.amax(next_q_values[0])

                target_vec = self.model.predict(state.reshape(1, -1), verbose=0)[0]
                target_vec[action] = target

                self.model.fit(state.reshape(1, -1), target_vec.reshape(1, -1), epochs=1, verbose=0)

                state = next_state
                total_reward += reward

                # Decay exploration
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

        # Save the model
        self.model.save('simple_network_healing_model.h5')

    def load_trained_model(self, model_path):
        """
        Load a pre-trained model
        """
        self.model = load_model(model_path, custom_objects={'mse': mse})

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
