import math
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

        if action == 0:
            self.owned_capital[time + 1] =\
                self.rate_jpy_dollar[time + 1] / \
                    self.rate_jpy_dollar[time] * self.owned_capital[time]
        if action == 1:
            self.owned_capital[time + 1] =\
                (1 - (self.rate_jpy_dollar[time + 1] / \
                    self.rate_jpy_dollar[time])) * self.owned_capital[time]
        if action == 2:
            self.owned_capital[time + 1] = self.owned_capital[time]
        
        print(self.owned_capital)
        reward = math.log(self.owned_capital[time + 1] / self.owned_capital[time])
        print(action)
        print(reward)
        return reward