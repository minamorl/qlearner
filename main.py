from enum import Enum
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas_datareader as web
import datetime
import random
from scipy.ndimage.interpolation import shift
from learner.constant import Action
from learner.agent import Agent
from learner.environment import Environment
from learner.error import StopLearningIteration
from learner.state import State


def calcrate_rcp_rmsd(dataset, relative_count):
    rcp_value = np.zeros_like(dataset)
    rmsd_value = np.zeros_like(dataset)
    for t in range(dataset.shape[0]):
        # Calcuration of RCP (relative closing place)
        mu = 1.0 / relative_count * \
            sum(dataset[t] for _ in range(t - relative_count + 1, t))
        sigma = 1.0 / relative_count * \
            sum(dataset[t] ** 2 - mu **
                2 for _ in range(t - relative_count + 1, t))
        rcp = (dataset[t] - mu) / (4 * sigma)

        # Calcuration of RMSD (relative moving standard d)
        mu2 = 1.0 / relative_count * \
            sum(sigma for _ in range(t - relative_count + 1, t))
        sigma2 = 1.0 / relative_count * \
            sum(sigma ** 2 for _ in range(t - relative_count + 1)) - mu2 ** 2
        rmsd = (sigma - mu2) / (4 * sigma2)

        rcp_value[t] = rcp
        rmsd_value[t] = rmsd
    return rcp_value, rmsd_value


def find_nerest(A, value):
    idx = (np.abs(A - value)).argmin()
    return A[idx]


def main():

    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    dataset = web.DataReader("DEXJPUS", "fred", start,  end)[
        'DEXJPUS'].dropna()
    owned_capital = np.ones((dataset.shape[0], ))
    rcp, rmsd = calcrate_rcp_rmsd(dataset, 50)

    rcp_min = rcp.min()
    rcp_max = rcp.max()
    rmsd_min = rmsd.min()
    rmsd_max = rmsd.max()
    grid_size = 5
    grid_rcp = np.linspace(rcp_min, rcp_max, num=grid_size + 1)
    grid_rmsd = np.linspace(rmsd_min, rmsd_max, num=grid_size + 1)

    def grid_state(state): return State(find_nerest(
        grid_rcp, state.rcp), find_nerest(grid_rmsd, state.rmsd))

    learning_times = 10000

    agent = Agent(step_size=1, discount_factor=0.9, investment_ratio=0.5,
                  q_value=np.zeros((grid_size * grid_size, 3)), eps=0.05)
    environment = Environment(
        rate_jpy_dollar=dataset,
        owned_capital=owned_capital,
        rcp=rcp,
        rmsd=rmsd,
    )

    for i in range(learning_times):
        time = random.randrange(dataset.shape[0])
        for j in range(time, dataset.shape[0] - 1):
            state = grid_state(environment.observe_state(time))
            print(state)
            action = agent.choose_action(state)
            reward = environment.apply_action(time, action)
            next_state = grid_state(environment.observe_state(time + 1))
            try:
                agent.update_q_value(action, state, next_state, reward)
            except StopLearningIteration as e:
                break
            environment.update_state(time)
            time = time + 1
        for x in agent.q_value:
            print(x)


main()
