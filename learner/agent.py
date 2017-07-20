from .constant import Action
import numpy as np
import math

class Agent:
    def __init__(self,
                 *,
                 step_size,
                 investment_ratio,
                 discount_factor,
                 q_value):
        self.step_size = step_size
        self.investment_ratio = investment_ratio
        self.discount_factor = discount_factor
        self.q_value = q_value

    def choose_action(self, time):
        """" Choose an action from q-value. Returns action."""
        return np.argmax(self.q_value[time])

    def update_q_value(self, time, action, reward):
        """ Update q-value """
        self.q_value[time][action] = \
            self.q_value[time][action] + \
            self.step_size * (math.log(1 + reward * self.investment_ratio) +\
            self.discount_factor * max(self.q_value[time + 1][i] for i in range(len(Action))) -\
            self.q_value[time][action])
