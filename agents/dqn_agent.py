"""Deep Q-Network Agent for Continuous State Space.

Implements DQN with experience replay and target network for the
autonomous delivery robot task with continuous state representation.
"""

from agents import BaseAgent
from collections import deque
import numpy as np
import random


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition to replay buffer.
        
        Args:
            state: Current state vector.
            action: Action taken.
            reward: Reward received.
            next_state: Next state vector.
            done: Whether episode terminated. 
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample random batch from replay buffer.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
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
    """Simple multi-layer perceptron for DQN approximation.
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 output_size: int = 4):
        """Initialize DQN network.
        
        Args:
            input_size: Size of state vector (default: 10 for continuous env).
            hidden_size: Size of hidden layers.
            output_size: Number of actions.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases for 3-layer MLP
        self.W1 = np.random.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
                
        self.W2 = np.random.normal(0, np.sqrt(2 / hidden_size), (hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        
        self.W3 = np.random.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b3 = np.zeros((1, output_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network.
        
        Args:
            x: Input state vector or batch of states.
            
        Returns:
            Q-values for each action.
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
        
        Args:
            x: Input state vector or batch.
            td_target: Target Q-values from Bellman equation.
            learning_rate: Learning rate for weight updates.
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
        
        # Update weights and biases
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def copy_from(self, other: 'DQNNetwork'):
        """Copy weights from another network (for target network).
        
        Args:
            other: Network to copy from.
        """
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for continuous state delivery robot task."""
    
    def __init__(self, n_actions: int = 4, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 min_epsilon: float = 0.01, epsilon_anneal_steps: int = 100000,
                 replay_capacity: int = 10000, batch_size: int = 32,
                 target_update_freq: int = 1000, input_size: int = 10,
                 hidden_size: int = 128,
                 reward_min: float = -10.0, reward_max: float = float('inf')):
        """Initialize DQN Agent.

        Args:
            n_actions: Number of discrete actions.
            learning_rate: Learning rate for network updates.
            gamma: Discount factor.
            epsilon: Initial exploration rate.
            min_epsilon: Minimum exploration rate (also used as anneal target).
            epsilon_anneal_steps: Number of training steps to linearly anneal epsilon.
            replay_capacity: Size of replay buffer.
            batch_size: Batch size for training.
            target_update_freq: Steps between target network updates.
            input_size: Size of state vector.
            hidden_size: Size of hidden layers in network.
            reward_min: Lower bound for reward clipping (default -10 covers collision penalty).
            reward_max: Upper bound for reward clipping (default inf = no upper clipping,
                preserving the large goal reward 10*W*H).
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
        
        # Warm-up: don't train until buffer has this many transitions
        self.warmup = min(replay_capacity, 10000)
        # Epsilon annealing (linear): track start/end and anneal duration (in training steps)
        self.epsilon_start = epsilon
        self.epsilon_end = min_epsilon
        self.epsilon_anneal_steps = epsilon_anneal_steps
        # Reward clipping
        self.reward_min = reward_min
        self.reward_max = reward_max
        # Logging
        self.log_every = 1000
        self.last_loss = None

        # Training tracking
        self.training_step = 0
        self.training = True
    
    def set_training(self, training: bool):
        """Set training mode.
        
        Args:
            training: Whether agent is in training mode.
        """
        self.training = training
    
    def greedy_action(self, state: np.ndarray) -> int:
        """Select action greedily using current Q-network.
        
        Args:
            state: Current state vector.
            
        Returns:
            Action with highest Q-value.
        """
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)
    
    def take_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy.
        
        Args:
            state: Current state vector.
            
        Returns:
            Action to take.
        """
        if self.training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        return self.greedy_action(state)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store transition and update Q-network.
        
        Args:
            state: Current state vector.
            action: Action taken.
            reward: Reward received.
            next_state: Next state vector.
            done: Whether episode terminated.
        """
        # Clip reward and store in replay buffer
        reward = float(np.clip(reward, self.reward_min, self.reward_max))
        self.replay_buffer.push(state, action, reward, next_state, done)

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
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute Q-values for next states using target network
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        
        # Compute target Q-values using Bellman equation
        td_target = self.q_network.forward(states)
        batch_len = states.shape[0]
        for i in range(batch_len):
            if dones[i]:
                td_target[i, actions[i]] = rewards[i]
            else:
                td_target[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # Compute and store loss (MSE on taken actions) for monitoring
        q_pred = self.q_network.forward(states)
        idx = np.arange(batch_len)
        pred_q = q_pred[idx, actions]
        target_q = td_target[idx, actions]
        loss = np.mean((pred_q - target_q) ** 2)
        self.last_loss = float(loss)
        if self.training_step % self.log_every == 0:
            print(f"[DQN] step={self.training_step} loss={self.last_loss:.6f} buffer={len(self.replay_buffer)} epsilon={self.epsilon:.4f}")

        # Update Q-network
        self.q_network.backward(states, td_target, learning_rate=self.learning_rate)
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        # Keep epsilon consistent with linear annealing used during training steps.
        if self.training_step <= self.epsilon_anneal_steps:
            frac = float(self.training_step) / float(self.epsilon_anneal_steps)
            self.epsilon = max(self.min_epsilon,
                               self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac)
        else:
            self.epsilon = max(self.min_epsilon, self.epsilon_end)