import math
import numpy as np
from .constant import Action


class Environment:
    def __init__(self,
                 *,
                 rate_jpy_dollar,
                 owned_capital,
                 rcp,
                 rmsd
                 ):
        self.rate_jpy_dollar = rate_jpy_dollar
        self.owned_capital = owned_capital
        self.rcp = rcp
        self.rmsd = rmsd

    def observe_state(self, time):
        """ Returns environment state at time"""
        return (self.rcp[time], self.rmsd[time])

    def apply_action(self, time, action):
        """ Returns rewards """
        if math.isnan(self.rate_jpy_dollar[time + 1]):
            self.rate_jpy_dollar[time + 1] = self.rate_jpy_dollar[time]
            return 0

        if action == Action.LONG:
            self.owned_capital[time + 1] =\
                self.rate_jpy_dollar[time + 1] / \
                self.rate_jpy_dollar[time] * self.owned_capital[time]
        if action == Action.SHORT:
            self.owned_capital[time + 1] =\
                (2 - (self.rate_jpy_dollar[time + 1] /
                      self.rate_jpy_dollar[time])) * self.owned_capital[time]
        if action == Action.FLAT:
            self.owned_capital[time + 1] = self.owned_capital[time]
        reward = math.log(
            self.owned_capital[time + 1] / self.owned_capital[time])
        return reward

    def update_state(self, time):
        self.rcp[time + 1] = self.rcp[time]
        self.rmsd[time + 1] = self.rmsd[time]