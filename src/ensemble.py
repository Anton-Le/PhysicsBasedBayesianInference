import numpy as np
from scipy.stats import norm
from scipy.constants import k as boltzmannConst

class Ensemble( ):


    def __init__(self, numParticles, numDimensions, potential):
        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.m = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)
        self.potential = potential


    def initialize(self, mass, temperature, q_std):
        if len(mass) != numParticles or np.ndim != 1:
            raise ValueError('Mass must be 1D array of length numParticles.')

        self.q = norm.rvs(scale=q_std, size=(numDimensions, numParticles))
        p_std = np.sqrt(mass * boltzmannConst * temperature)
        self.p = norm.rvs(scale=scale, size=(numDimensions, numParticles))

        self.weights = np.exp(-potential())

    def __iter__(self):
        return self.q, self.p, self.m, self.weights, self.potential


    def particle(self, particleNum):
# I'm not sure if brackets follow correct style here - please correct if needed        
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.m[particleNum], self.weights[particleNum]
