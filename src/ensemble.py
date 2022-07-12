import numpy as np
from scipy.stats import norm
from scipy.constants import k as boltzmannConst

class Ensemble( ):


    def __init__(self, numParticles, numDimensions):
        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.mass = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)
        print('Warning: Mass initialized to 0.')

    def __iter__(self):
        return iter((self.q, self.p, self.mass, self.weights))

    def initialize(self, temperature):    
        std = np.sqrt(self.mass * boltzmannConst * temperature)
        self.p = norm.rvs(scale=std, size=)

    def particle(self, particleNum):
# I'm not sure if brackets follow correct style here - please correct if needed        
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.mass[particleNum], self.weights[particleNum]
