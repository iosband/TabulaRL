'''
Finite horizon RLSVI with fixed action choices.

author: iosband@stanford.edu

'''

import numpy as np
from agent import *

class FiniteHorizonRLSVI(FiniteHorizonAgent):
    '''
    FiniteHorizonRLSVI is RLSVI for finite horizon and fixed action choices.

    Important part is the memory, a list of lists.
        covs[h] = Sigma_h
        thetaMeans[h] = \overline{theta}_h
        thetaSamps[h] = \hat{theta}_h
        memory[h] = {oldFeat, reward, newFeat}
            oldFeat = A (nData x nFeat)
            reward = vector of rewards
            newFeat = array (nData x nFeat x nAction)
    with history appended row by row.
    '''
    def __init__(self, nFeat, nAction, epLen,
                 epsilon=0, sigma=1, lam=1, maxHist=1e6, discount=1):
        self.nFeat = nFeat
        self.nAction = nAction
        self.epLen = epLen
        self.epsilon = epsilon
        self.sigma = sigma
        self.maxHist = maxHist
        self.discount = discount

        # Make the computation structures
        self.covs = []
        self.thetaMeans = []
        self.thetaSamps = []
        self.memory = []
        for i in range(epLen + 1):
            self.covs.append(np.identity(nFeat, dtype=np.float16) / float(lam))
            self.thetaMeans.append(np.zeros(nFeat), dtype=np.float16)
            self.thetaSamps.append(np.zeros(nFeat), dtype=np.float16)
            self.memory.append(
            {'oldFeat': np.zeros([maxHist, nFeat], dtype=np.float16),
             'rewards': np.zeros(maxHist, dtype=np.float16),
             'newFeat': np.zeros([maxHist, nAction, nFeat], dtype=np.float16)}
             )

    def update_obs(self, ep, h, oldObs, reward, newObs):
        '''
        Take in an observed transition and add it to the memory.

        Args:
            ep - int - which episode
            h - int - timestep within episode
            oldObs - nFeat x 1
            action - int
            reward - float
            newObs - nFeat x nAction

        Returns:
            NULL - update covs, update memory in place.
        '''
        if ep >= self.maxHist:
            print('****** ERROR: Memory Exceeded ******')

        # Adding the memory
        self.memory[h]['oldFeat'][ep, :] = oldObs
        self.memory[h]['rewards'][ep] = reward
        self.memory[h]['newFeat'][ep, :, :] = newObs

        if len(self.memory[h]['oldFeat']) == len(self.memory[h]['rewards']) \
           and len(self.memory[h]['rewards']) == len(self.memory[h]['newFeat']):
            pass
        else:
            print('****** ERROR: Memory Failure ******')

    def update_policy(self, ep):
        '''
        Re-computes theta parameters via planning step.

        Args:
            ep - int - which episode are we on

        Returns:
            NULL - updates theta in place for policy
        '''
        H = self.epLen

        if len(self.memory[H - 1]['oldFeat']) == 0:
            return

        for i in range(H):
            h = H - i - 1
            A = self.memory[h]['oldFeat'][0:ep]
            nextPhi = self.memory[h]['newFeat'][0:ep, :, :]
            nextQ = np.dot(nextPhi, self.thetaSamps[h + 1])
            maxQ = nextQ.max(axis=1)
            b = self.memory[h]['rewards'][0:ep] + self.discount * maxQ

            self.thetaMeans[h] = \
                self.covs[h].dot(np.dot(A.T, b)) / (self.sigma ** 2)
            L = np.linalg.cholesky(self.covs[h])
            self.thetaSamps[h] = \
                self.thetaMeans[h] + np.dot(L, np.random.randn(self.nFeat))

    def pick_action(self, t, obs):
        '''
        The greedy policy according to thetaSamps

        Args:
            t - int - timestep within episode
            obs - nAction x nFeat - features for each action

        Returns:
            action - int - greedy with respect to thetaSamps
        '''
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.nAction)
        else:
            qVals = np.dot(self.thetaSamps[t], obs.T)
            return qVals.argmax()
