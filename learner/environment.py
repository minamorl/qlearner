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

        if action == Action.LONG:
            self.owned_capital[time + 1] =\
                self.rate_jpy_dollar[time ] / \
                    self.rate_jpy_dollar[time] * self.owned_capital
        if action == Action.SHORT:
            self.owned_capital[time + 1] =\
                1 - (self.rate_jpy_dollar[time ] / \
                    self.rate_jpy_dollar[time]) * self.owned_capital
        if action == Action.FLAT:
            self.owned_capital[time + 1] = self.owned_capital[time]

        reward = self.owned_capital[time + 1] - self.owned_capital[time]
        return reward