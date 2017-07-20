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

def main():

    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    dataset = web.DataReader("DEXJPUS", "fred", start,  end)['DEXJPUS']
    
    time = random.randrange(dataset.shape[0])
    owned_capital = np.ones((dataset.shape[0], )) * 10000
    agent = Agent(step_size=1, discount_factor=0.9, investment_ratio=2.0, q_value=np.zeros((dataset.shape[0], 3)))
    environment = Environment(
        rate_jpy_dollar=dataset,
        owned_capital=owned_capital)

    for i in range(time, dataset.shape[0] - 1):
        action = agent.choose_action(time)
        reward = environment.apply_action(time, action)
        agent.update_q_value(time, action, reward)
        time = time + 1
main()
