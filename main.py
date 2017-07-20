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
    agent = Agent(step_size=1.0, discount_factor=0.8, investment_ratio=1.0, q_value=np.zeros((dataset.shape[0], 3)))
    environment = Environment(
        rate_jpy_dollar=dataset, owned_capital=np.zeros((dataset.shape[0], )))

    for i in range(dataset.shape[0] - 1):
        action = agent.choose_action(time)
        reward = environment.apply_action(time, action)
        print(action)
        print(reward)
        agent.update_q_value(time, action, reward)
        time = time + 1
main()
