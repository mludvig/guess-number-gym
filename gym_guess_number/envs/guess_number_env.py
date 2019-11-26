#!/usr/bin/env python3

"""
Simple Guess a Number game

Observation is [NONE, LOWER, HIGHER, CORRECT]

Action space is [0 .. MAX_NUMBER]

The reward 0 until the last episode, then Action distance from Target number.
The distance is absolute and negative, ie the closer the last action to target the higher reward.
"""

# core modules
import logging.config
import math
import pkg_resources
import random
from enum import IntEnum

# 3rd party modules
from gym import spaces
import yaml
import gym
import numpy as np
import math

config_file = pkg_resources.resource_filename('gym_guess_number', 'config.yaml')
with open(config_file, 'rt') as f:
    config = yaml.load(f)
logging.config.dictConfig(config['LOGGING'])


class Observation(IntEnum):
    NONE = 0
    CORRECT = 1
    LOWER = 2
    HIGHER = 3

class GuessNumberEnv(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.0.1"
        logging.info(f"GuessNumberEnv - Version {self.__version__}")

        # General variables defining the environment
        self.MAX_NUMBER = 100

        # Give it enough tries to guess correctly with the right strategy
        self.MAX_STEPS = int(math.log2(self.MAX_NUMBER)) + 1

        # The numbers the agent can choose from (must be 'self.action_space')
        self.action_space = spaces.Discrete(self.MAX_NUMBER + 1)

        # Observation is the hint (Observation)
        #low = np.array([0.0,])
        #high = np.array([4.0,])
        #self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Box(
                low=np.array([min(Observation)]),
                high=np.array([max(Observation)]),
                dtype=np.int32)

        #self.action_episode_memory = []

        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        #self.action_episode_memory.append([])
        self.is_over = False
        self.steps_remaining = self.MAX_STEPS
        self.target_number = random.randint(0, self.MAX_NUMBER)
        self.info = {'t': self.target_number, 'a': []}
        return [ Observation.NONE ]

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
        """
        if self.is_over:
            raise RuntimeError("Episode is done")
        #self.action_episode_memory[-1].append(action)
        self.info['a'].append(action)
        self.steps_remaining -= 1
        if self.steps_remaining == 0 or action == self.target_number:
            self.is_over = True
            reward = -abs(action-self.target_number)
        else:
            reward = -0.1  # Don't give reward until the last step

        if action < self.target_number:
            ob = [ Observation.HIGHER ]
        elif action > self.target_number:
            ob = [ Observation.LOWER ]
        else:
            ob = [ Observation.CORRECT ]

        return ob, reward, self.is_over, self.info

    def _render(self, mode='human', close=False):
        return

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
