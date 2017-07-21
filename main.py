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


def main():

    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    dataset = web.DataReader("DEXJPUS", "fred", start,  end)['DEXJPUS']

    owned_capital = np.ones((dataset.shape[0], )) * 100
    agent = Agent(step_size=1, discount_factor=0.9, investment_ratio=0.5,
                  q_value=np.zeros((dataset.shape[0], 3)), eps=0.05)
    environment = Environment(
        rate_jpy_dollar=dataset,
        owned_capital=owned_capital)
    
    learning_times = 10000

    for i in range(learning_times):
        time = random.randrange(dataset.shape[0])
        for j in range(time, dataset.shape[0] - 1):
            action = agent.choose_action(time)
            reward = environment.apply_action(time, action)
            try:
                agent.update_q_value(time, action, reward)
            except StopLearningIteration as e:
                break
            time = time + 1
        print(agent.q_value)


main()
