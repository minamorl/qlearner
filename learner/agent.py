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
        self.eps = eps

    def choose_action(self, time):
        """" Choose an action from q-value. Returns action."""
        rate = random.random()
        if rate < 1.0 - self.eps:
            argmax = np.argmax(self.q_value[time])
            return argmax
        return random.randrange(len(Action))
        
    def update_q_value(self, time, action, reward):
        """ Update q-value """
        next_q_value = \
            self.q_value[time][action] + \
            self.step_size * (
            math.log(1 + (reward - 1) * self.investment_ratio) +\
            self.discount_factor * max(self.q_value[time + 1][i] for i in range(len(Action))) -\
            self.q_value[time][action])
        if not self.q_value[time][action]:
            self.q_value[time][action] = next_q_value
            return
        raise StopLearningIteration()