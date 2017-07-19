from enum import Enum
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas_datareader as web
import datetime
import random
from scipy.ndimage.interpolation import shift

class IGeneralLearner(metaclass=ABCMeta):
    @abstractmethod
    def create(self):
        """Create an initialized lerner instance"""
        pass

class QLearner(IGeneralLearner):
    """A class for Q-learning."""

    def __init__(self,
            *, 
            _internal_call=False,
            learning_rate,
            discount_factor,
            q_value,
            actions,
            rewards,
            states
            ):
        if not _internal_call:
            raise ValueError("Do not call the consturctor directly. Use create() instead.")

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_value = q_value
        self.actions = actions
        self.rewards = rewards
        self.states = states

    def initialize_state(self):
        """ Pick a random state """
        return random.randrange(self.q_value.shape[1])

    def find_action(self, state):
        """ Find an action from highest q-value. """
        return np.argmax(self.q_value[state])

    def calcurate_next_q_value(self, state, action):
        reward = self.rewards[state][action]
        return self.q_value[state][action] + self.learning_rate * (
            reward + self.discount_factor * 
            max([self.q_value[state][i] for i in self.actions.shape[0]])
            - self.q_value[state][action])

    def learn(self, dataset, learning_count):
        state = self.initialize_state()
        for i in range(learning_count):
            action = self.find_action(state)
            print(self.q_value)
            self.q_value[state][action] = self.calcurate_next_q_value(state, action)
            state = state + 1
            if state == self.actions.shape[0] - 1:
                break

    @classmethod
    def create(cls, **options):
        """Create an initialized QLearner instance."""
        instance = cls(
            _internal_call=True,
            **cls._initialize_options(options))
        return instance

    @classmethod
    def _initialize_options(cls, options):
        """An internal function for intialization."""
        return options

def main():
    
    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    dataset = web.DataReader("DEXJPUS", "fred", start,  end)['DEXJPUS']
    learner = QLearner.create(
        learning_rate=1.0,
        discount_factor=0.8,
        q_value=np.zeros((15, 3)),
        actions=np.array([0, 1, 2]),
        rewards=np.zeros((15, 3))
    )
#    learner.learn(100)

main()