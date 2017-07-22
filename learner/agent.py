from .constant import Action
import numpy as np
import math
import random
from learner.error import StopLearningIteration


class Agent:
    def __init__(self,
                 *,
                 step_size,
                 investment_ratio,
                 discount_factor,
                 q_value,
                 eps):
        self.step_size = step_size
        self.investment_ratio = investment_ratio
        self.discount_factor = discount_factor
        self.q_value = q_value
        self.map_state_to_q_value_index = dict()
        self.eps = eps

    def choose_action(self, state):
        """" Choose an action from q-value. Returns action."""
        rate = random.random()
        if rate < 1.0 - self.eps:
            argmax = np.argmax(self.q_value[self._find_q_value_index(state)])
            return argmax
        return random.randrange(len(Action))

    def _find_q_value_index(self, state):
        """ Find index of q-value. If given state is not in q_value_keys, append it and return new index. """
        if state in self.map_state_to_q_value_index.keys():
            return self.map_state_to_q_value_index[state]
        index = len(self.map_state_to_q_value_index)
        self.map_state_to_q_value_index[state] = index
        return index

    def update_q_value(self, action, state, next_state, reward):
        """ Update q-value """
        q_value_index = self._find_q_value_index(state)
        next_q_value_index = self._find_q_value_index(next_state)
        next_q_value = \
            self.q_value[q_value_index][action] + \
            self.step_size * (
                math.log(1 + (reward - 1) * self.investment_ratio) +
                self.discount_factor * max(self.q_value[next_q_value_index][i] for i in range(len(Action))) -
                self.q_value[q_value_index][action])
        self.q_value[q_value_index][action] = next_q_value    
