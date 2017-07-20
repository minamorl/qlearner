import math
import numpy as np
from .constant import Action

class Environment:
    def __init__(self,
                 *,
                 rate_jpy_dollar,
                 owned_capital,
                 ):
        self.rate_jpy_dollar = rate_jpy_dollar
        self.owned_capital = owned_capital

    def observe_state(self, time):
        """ Returns environment state at time"""
        return self.rate_jpy_dollar[time]

    def apply_action(self, time, action):
        """ Returns rewards """
        if self.rate_jpy_dollar[time] == np.nan or self.rate_jpy_dollar[time + 1] == np.nan:            
            self.owned_capital[time + 1] = self.owned_capital[time]

        if action == 0:
            self.owned_capital[time + 1] =\
                self.rate_jpy_dollar[time + 1] / \
                    self.rate_jpy_dollar[time] * self.owned_capital[time]
        if action == 1:
            self.owned_capital[time + 1] =\
                (2 - (self.rate_jpy_dollar[time + 1] / self.rate_jpy_dollar[time])) * self.owned_capital[time]
        if action == 2:
            self.owned_capital[time + 1] = self.owned_capital[time]
        
        reward = math.log(self.owned_capital[time + 1] / self.owned_capital[time])
        return reward