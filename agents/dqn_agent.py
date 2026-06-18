"""Deep Q-Network Agent for Continuous State Space.

Implements DQN with experience replay and target network for the continuous state representation.
"""

from agents import BaseAgent
from collections import deque
import numpy as np
import random
import torch


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.

        :param capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition to replay buffer.

        :param state: Current state vector.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state vector.
        :param done: Whether episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample random batch from replay buffer.

        :param batch_size: Number of transitions to sample.
        :return: Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        transitions = random.sample(self.buffer, batch_size)

        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.array([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNNetwork:
    """Simple multi-layer perceptron for DQN approximation."""

    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 output_size: int = 24):
        """Initialize DQN network.

        :param input_size: Size of state vector (default: 10 for continuous env).
        :param hidden_size: Size of hidden layers.
        :param output_size: Number of actions.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He initialisation for ReLU networks
        self.W1 = np.random.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.normal(0, np.sqrt(2 / hidden_size), (hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = np.random.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b3 = np.zeros((1, output_size))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network.

        :param x: Input state vector or batch of states.
        :return: Q-values for each action.
        """
        # Ensure 2D shape for batch operations
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # First hidden layer with ReLU
        self.h1 = np.maximum(0, np.dot(x, self.W1) + self.b1)

        # Second hidden layer with ReLU
        self.h2 = np.maximum(0, np.dot(self.h1, self.W2) + self.b2)

        # Output layer (linear)
        q_values = np.dot(self.h2, self.W3) + self.b3

        return q_values

    def backward(self, x: np.ndarray, td_target: np.ndarray, learning_rate: float = 0.001):
        """Backward pass to update weights via gradient descent.

        :param x: Input state vector or batch.
        :param td_target: Target Q-values from Bellman equation.
        :param learning_rate: Learning rate for weight updates.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch_size = x.shape[0]

        # Forward pass
        q_pred = self.forward(x)

        # Compute loss gradient
        dq = (q_pred - td_target) / batch_size

        # Backpropagation through output layer
        dW3 = np.dot(self.h2.T, dq)
        db3 = np.sum(dq, axis=0, keepdims=True)

        # Backprop through second hidden layer
        dh2 = np.dot(dq, self.W3.T)
        dh2[self.h2 <= 0] = 0  # ReLU gradient
        dW2 = np.dot(self.h1.T, dh2)
        db2 = np.sum(dh2, axis=0, keepdims=True)

        # Backprop through first hidden layer
        dh1 = np.dot(dh2, self.W2.T)
        dh1[self.h1 <= 0] = 0  # ReLU gradient
        dW1 = np.dot(x.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)

        # Clip gradients to prevent exploding weights.
        # Must be large enough to allow Q-values to grow toward the target reward
        # So increased this, so should be better like this. 
        # TODO: Might be something that can be tuned a bit further
        clip = 5.0
        dW3, db3 = np.clip(dW3, -clip, clip), np.clip(db3, -clip, clip)
        dW2, db2 = np.clip(dW2, -clip, clip), np.clip(db2, -clip, clip)
        dW1, db1 = np.clip(dW1, -clip, clip), np.clip(db1, -clip, clip)

        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def copy_from(self, other: 'DQNNetwork'):
        """Copy weights from another network (for target network).

        :param other: Network to copy from.
        """
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for continuous state delivery robot task."""

    def __init__(self, n_actions: int = 24, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 min_epsilon: float = 0.01, epsilon_anneal_steps: int = 100000,
                 replay_capacity: int = 10000, batch_size: int = 32,
                 target_update_freq: int = 1000, input_size: int = 10,
                 hidden_size: int = 128):
        """Initialize DQN Agent.

        :param n_actions: Number of discrete actions.
        :param learning_rate: Learning rate for network updates.
        :param gamma: Discount factor.
        :param epsilon: Initial exploration rate.
        :param min_epsilon: Minimum exploration rate (also used as anneal target).
        :param epsilon_anneal_steps: Number of training steps to linearly anneal epsilon.
        :param replay_capacity: Size of replay buffer.
        :param batch_size: Batch size for training.
        :param target_update_freq: Steps between target network updates.
        :param input_size: Size of state vector.
        :param hidden_size: Size of hidden layers in network.
        """
        # Hyperparameters
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = DQNNetwork(input_size, hidden_size, n_actions)
        self.target_network = DQNNetwork(input_size, hidden_size, n_actions)
        self.target_network.copy_from(self.q_network)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Warm-up: don't train until buffer has this many transitions.
        # Made a bit smaller relative to replay capacity so training starts a bit earlier
        self.warmup = min(replay_capacity // 2, 10000)
        # Epsilon annealing (linear): track start/end and anneal duration (in training steps)
        self.epsilon_start = epsilon
        self.epsilon_end = min_epsilon
        self.epsilon_anneal_steps = epsilon_anneal_steps
        # Logging
        self.log_every = 1000
        self.last_loss = None

        # Training tracking
        self.training_step = 0
        self.training = True

    def set_training(self, training: bool):
        """Set training mode.

        :param training: Whether agent is in training mode.
        """
        self.training = training

    def greedy_action(self, state: np.ndarray) -> int:
        """Select action greedily using current Q-network.

        :param state: Current state vector.
        :return: Action with highest Q-value.
        """
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)

    def take_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy.

        :param state: Current state vector.
        :return: Action to take.
        """
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        return self.greedy_action(state)

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store transition and update Q-network.

        :param state: Current state vector.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state vector.
        :param done: Whether episode terminated.
        """
        self.replay_buffer.push(state, action, float(reward), next_state, done)

        # Only train after warm-up and when we have at least one full batch
        if len(self.replay_buffer) >= max(self.batch_size, self.warmup) and self.training:
            self._train_step()
            self.training_step += 1

            # Linear epsilon annealing based on training steps
            if self.training_step <= self.epsilon_anneal_steps:
                frac = float(self.training_step) / float(self.epsilon_anneal_steps)
                self.epsilon = max(self.min_epsilon,
                                   self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac)

            # Update target network
            if self.training_step % self.target_update_freq == 0:
                self.target_network.copy_from(self.q_network)

    def _train_step(self):
        """Perform single training step on batch from replay buffer."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        max_next_q = np.max(self.target_network.forward(next_states), axis=1)

        td_target = self.q_network.forward(states)
        idx = np.arange(states.shape[0])
        orig_q = td_target[idx, actions].copy()

        td_target[idx, actions] = rewards + self.gamma * max_next_q * (1.0 - dones.astype(float))

        self.last_loss = float(np.mean((orig_q - td_target[idx, actions]) ** 2))
        # if self.training_step % self.log_every == 0:
        #     print(f"[DQN] step={self.training_step} loss={self.last_loss:.6f} buffer={len(self.replay_buffer)} epsilon={self.epsilon:.4f}")

        self.q_network.backward(states, td_target, learning_rate=self.learning_rate)

    def save(self, path, obs_mode: str = "both") -> None:
        """Save q_network weights, and obs_mode to a .pt file."""
        torch.save({
            "agent_type": "dqn",
            "obs_mode":   obs_mode,
            "W1": self.q_network.W1, "b1": self.q_network.b1,
            "W2": self.q_network.W2, "b2": self.q_network.b2,
            "W3": self.q_network.W3, "b3": self.q_network.b3,
        }, path)

    @classmethod
    def load(cls, path) -> 'DQNAgent':
        """Load a DQNAgent"""
        checkpoint = torch.load(path, weights_only=False)
        agent = cls(
            n_actions   = checkpoint["W3"].shape[1],
            input_size  = checkpoint["W1"].shape[0],
            hidden_size = checkpoint["W1"].shape[1],
        )
        agent.q_network.W1 = checkpoint["W1"]
        agent.q_network.b1 = checkpoint["b1"]
        agent.q_network.W2 = checkpoint["W2"]
        agent.q_network.b2 = checkpoint["b2"]
        agent.q_network.W3 = checkpoint["W3"]
        agent.q_network.b3 = checkpoint["b3"]
        agent.obs_mode = checkpoint["obs_mode"]
        agent.set_training(False)
        return agent
