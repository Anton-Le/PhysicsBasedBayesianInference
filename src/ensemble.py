#!/usr/bin/env python3 

import numpy as np
from scipy.stats import norm
from scipy.constants import k as boltzmannConst

class Ensemble( ):
    ''' Class containing particle positions, momenta and masses as well as weights and 
    a potential taking only position as an argument.
    '''
    potential = None # If None integrator runs N-body simulation, 
                    # use potential.noPotential for free particles

    def __init__(self, numDimensions, numParticles, potential):

    # If potential = None integrator runs N-body simulation.

        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.mass = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)
        self.potential = potential

    def __iter__(self):
        ''' Allows for easy unpacking of ensemble object.
        '''
        return self.q, self.p, self.mass, self.weights, self.potential

    def setWeights(self, temperature):
        kineticEnergy = np.sum((self.p ** 2 / (2 * self.mass)), axis=0)
        hamiltonian = self.potential(self.q) + kineticEnergy
        self.weights = np.exp(- hamiltonian / (boltzmannConst * temperature))

    def initializeThermal(self, mass, temperature, q_std):
        ''' Distribute momentum based on a thermal distribution and position
        with a normal distribution.

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


    def particle(self, particleNum):
        ''' Return information about the particleNum th particle.
        '''      
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.mass[particleNum], self.weights[particleNum]
