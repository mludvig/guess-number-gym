#!/usr/bin/env python3

"""
Simple Guess a Number game

Observation is [NONE, LOWER, HIGHER, CORRECT]

Action space is [0 .. MAX_NUMBER]

The reward 0 until the last episode, then Action distance from Target number.
The distance is absolute and negative, ie the closer the last action to target the higher reward.
"""

import logging.config
import math
import random
from enum import IntEnum

import yaml
import pkg_resources

import gym
from gym import spaces
import numpy as np

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

    __version__ = "0.0.1"

    def __init__(self):
        logging.info("GuessNumberEnv - Version %s", self.__version__)

        # General variables defining the environment
        self.MAX_NUMBER = 100

        # The numbers the agent can choose from (must be 'self.action_space')
        self.action_space = spaces.Discrete(self.MAX_NUMBER + 1)

        # Observation is what we return back to the agent
        self.observation_space = spaces.Box(
            low=np.array([min(Observation)]),
            high=np.array([max(Observation)]),
            dtype=np.int32)

        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        # Give it enough tries to guess correctly with the right strategy
        # that is by halving the interval each time.
        self.steps_remaining = int(math.log2(self.MAX_NUMBER)) + 1

        self.target_number = random.randint(0, self.MAX_NUMBER)

        # Interval where the action should fall for extra reward - updated at every step
        self.interval = [ 0, self.MAX_NUMBER ]

        self.is_over = False
        self.info = {'t': self.target_number, 'a': []}
        self.observation = [Observation.NONE]
        logging.debug("reset() -> {}".format((self.observation, None, None, self.info)))
        return self.observation

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

        self.info['a'].append(action)

        # Current observation
        prev_observation = self.observation[0]
        if action < self.target_number:
            self.observation = [Observation.HIGHER]
            self.interval[0] = max(action, self.interval[0])          # update lower bound - target is higher
        elif action > self.target_number:
            self.observation = [Observation.LOWER]
            self.interval[1] = min(action, self.interval[1])          # update upper bound - target is lower
        else:
            self.observation = [Observation.CORRECT]

        self.info['i'] = self.interval

        # Is the episode over?
        self.steps_remaining -= 1
        if self.steps_remaining == 0 or action == self.target_number:
            self.is_over = True
            reward = -abs(action-self.target_number)    # Reward is negative distance from last action
        elif prev_observation == Observation.NONE:
            # If it's the first try reward is 0
            reward = 0.0
        else:
            # We already gave it a Lower/Higher hint - did respond correctly?
            prev_action = self.info['a'][-2]
            if ((prev_observation == Observation.HIGHER and action > prev_action) or
                (prev_observation == Observation.LOWER and action < prev_action)):
                # Yes it understands Lower/Higher
                reward = 0.1

                # Did it guess from the correct interval?
                if action in range(self.interval[0], self.interval[1]+1):
                    # Yes, much higher reward
                    reward = 0.5
            else:
                reward = -0.5

        ret = (self.observation, reward, self.is_over, self.info)
        logging.debug("step({}) -> {}".format(action, ret))
        return ret

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Optionally seed the RNG to get predictable results.

        Parameters
        ----------
        seed : int or None
        """
        random.seed(seed)
        np.random.seed(seed)
