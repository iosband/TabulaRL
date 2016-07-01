'''
Feature extractor to get features from an environment.
Currently only works with tabular MDPs.

author: iosband@stanford.edu
'''

import numpy as np
from environment import TabularMDP

#-------------------------------------------------------------------------------
class FeatureExtract(object):
    '''Get features out of an environment'''

    def __init__(self, epLen, nState, nAction, nFeat):
        self.epLen = epLen
        self.nState = nState
        self.nAction = nAction
        self.nFeat = nFeat

    def get_feat(self, env):
        '''Get the features out of the environment'''

    def check_env(self, env):
        ''' Check if a feature extractor is compatible with an environment '''
        assert(self.epLen == env.epLen)
        assert(self.nState == env.nState)
        assert(self.nAction == env.nAction)

    def check_agent(self, agent):
        ''' Check if a feature extractor is compatible with an environment '''
        assert(self.epLen == agent.epLen)
        assert(self.nFeat == agent.nFeat)
        assert(self.nAction == agent.nAction)


#----------------------------------------------------------------------------
class FeatureTrueState(FeatureExtract):
    '''Trivial feature extractor which just gives the state'''

    def get_feat(self, env):
        '''
        Args:
            env - TabularMDP

        Returns:
            timestep - int - timestep within episode
            state - int - state of the environment
        '''
        return env.timestep, env.state

#----------------------------------------------------------------------------
class LookupFeatureExtract(FeatureExtract):
    '''Simple lookup phi feature extractor'''

    def __init__(self, epLen, nState, nAction, nFeat):
        '''Very simple implementation, lookup Phi for now'''
        self.epLen = epLen
        self.nState = nState
        self.nAction = nAction
        self.nFeat = nFeat

        self.phi = np.zeros((epLen, nState, nAction, nFeat))

    def get_feat(self, env):
        '''
        Get all the state features for an environment.

        Args:
            env - Tabular MDP environment

        Returns:
            phi(h, s, :, :) - nAction x nFeat - array of features
        '''
        return self.phi[env.timestep, env.state, :, :]

    def check_env(self, env):
        ''' Check if a feature extractor is compatible with an environment '''
        assert(self.epLen == env.epLen)
        assert(self.nState == env.nState)
        assert(self.nAction == env.nAction)

    def check_agent(self, agent):
        ''' Check if a feature extractor is compatible with an environment '''
        assert(self.epLen == agent.epLen)
        assert(self.nFeat == agent.nFeat)
        assert(self.nAction == agent.nAction)
