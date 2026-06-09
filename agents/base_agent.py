"""Agent Base.

We define the base class for all agents in this file.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """

    @abstractmethod
    def take_action(self, state: tuple[float, float]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        raise NotImplementedError
    
    @abstractmethod
    # DQN & PPO require next_state to compute the bootstrap target using the Bellman equation: Q_target = reward + γ * max_a' Q_network(next_state)
    # DQN & PPO also require done to know when to stop bootstrapping (i.e., if done is True, then Q_target = reward)
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool): 
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        raise NotImplementedError
