import numpy as np
import random
import pickle


class QLearningAgent:
    def __init__(
        self,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # State: [Lane(2), Safe_Front(2), Safe_Left(2), Safe_Right(2)]
        # Shape: (2, 2, 2, 2, action_size)
        self.q_table = np.zeros((2, 2, 2, 2, action_size))

    def get_action(self, state):
        # State is np array [lane, safe_f, safe_l, safe_r]
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        idx = tuple(state)
        return np.argmax(self.q_table[idx])

    def learn(self, state, action, reward, next_state, done):
        idx = tuple(state)
        next_idx = tuple(next_state)

        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.q_table[next_idx])

        self.q_table[idx][action] += self.lr * (target - self.q_table[idx][action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
