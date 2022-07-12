import numpy as np
from scipy.stats import norm
from scipy.constants import k as boltzmannConst

class Ensemble( ):
    ''' Class containing particle positions and momenta, as well as weights and 
    a potential taking only position as an argument.
    '''

    def __init__(self, numDimensions, numParticles, potential):
        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.mass = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)
        self.potential = potential


    def setWeights(self, temperature):
        hamiltonian = self.potential(self.q) + self.p ** 2 / (2 * self.mass)
        self.weights = np.exp(- hamiltonian / boltzmannConst * temperature)

    def initialize(self, mass, temperature, q_std):
        ''' Distribute momentum based on a thermal distribution.
        

        '''
        if len(mass) != self.numParticles or np.ndim(mass) != 1:
            raise ValueError('Mass must be 1D array of length numParticles.')

        self.mass = mass
        
        self.q = norm.rvs(scale=q_std,
            size=(self.numDimensions, self.numParticles))
        
        # thermal distribution
        p_std = np.sqrt(self.mass * boltzmannConst * temperature)
        self.p = norm.rvs(scale=p_std, size=(self.numDimensions,
            self.numParticles))

        # set weights
        self.setWeights(temperature)
        

    def __iter__(self):
        return self.q, self.p, self.mass, self.weights, self.potential


    def particle(self, particleNum):
# I'm not sure if brackets follow correct style here - please correct if needed        
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.mass[particleNum], self.weights[particleNum]
