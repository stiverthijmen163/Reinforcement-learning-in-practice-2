"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
from agents.base_agent import BaseAgent
from world.helpers import *


class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def update(self, state: tuple[float, float], reward: float, action):
        pass

    def take_action(self, state: tuple[float, float]) -> int:
        return randint(0, len(ACTIONS) - 1)