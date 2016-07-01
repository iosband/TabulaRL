'''
All agents should inherit from the Agent class.

There are three common settings which we will examine:
- FiniteHorizonAgent = finite *known* horizon H
- EpisodicAgent = time-homogeneous problem with *unknown* episode length
- DiscountedAgent = infinite horizon with discount factor

Most work is presented for the FiniteHorizonAgent.

author: iosband@stanford.edu
'''

import numpy as np

class Agent(object):

    def __init__(self):
        pass

    def update_obs(self, obs, action, reward, newObs):
        '''Add observation to records'''

    def update_policy(self, h):
        '''Update internal policy based upon records'''

    def pick_action(self, obs):
        '''Select an observation based upon the observation'''


class FiniteHorizonAgent(Agent):
    pass

class EpisodicAgent(Agent):
    pass

class DiscountedAgent(Agent):
    pass
