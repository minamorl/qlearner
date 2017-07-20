from enum import Enum
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas_datareader as web
import datetime
import random
from scipy.ndimage.interpolation import shift
from .constant import Action
from .agent import Agent
from .environment import Environment

def main():

    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime(2015, 12, 31)
    dataset = web.DataReader("DEXJPUS", "fred", start,  end)['DEXJPUS']
    

main()
