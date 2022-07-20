#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: thomas

Contains Ensemble class.

"""

import numpy as np
from potential import getForce
from scipy.stats import norm
from scipy.constants import k as boltzmannConst

class Ensemble( ):
    """
    @description:
        Data structure containing information of an ensemble.
        Contains positions, momenta, masses, probabilistic weights of each
        particle, as well as the potential function.
    """

    def __init__(self, numDimensions, numParticles, potential):
        """
        @description:
            Initialize ensemble object
            
        @parameters:        
            numDimensions (int)
            numParticles (int)
            potential (func): Function taking only q as argument returning array of
            length nparticles of potentials.
        """
        # If potential = None integrator runs N-body simulation.

        
        self.numParticles = numParticles
        self.numDimensions = numDimensions
        self.q = np.zeros((numDimensions, numParticles))
        self.p = np.zeros((numDimensions, numParticles))
        self.mass = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)
        self.potential = potential
        self.dq = 1e-8 # for differentiation

    def __iter__(self):
        '''
        @description:
            Allows for easy unpacking of ensemble object.
        '''
        return self.q, self.p, self.mass, self.weights, self.potential

    def getAccel(self):
        """
        @description:
            Get acceleration at each position.
        
        @parameters:        
            self.q (ndarray): numDimensions x numParticles array
            self.mass (ndarray):
            self.potential (func):
            self.dq (float):
        """
        return getForce(self.q, self.potential, self.dq) / self.mass


    def setWeights(self, temperature):
        """
        @description:
            Set probabilistic weights.
         @parameters:        
            temperature (float):tests f
        """
        kineticEnergy = np.sum((self.p ** 2 / (2 * self.mass)), axis=0)
        hamiltonian = self.potential(self.q) + kineticEnergy
        self.weights = np.exp(- hamiltonian / (boltzmannConst * temperature))

    def initializeThermal(self, mass, temperature, q_std):
        """
        @description:
            Distribute momentum based on a thermal distribution and position
            with a normal distribution. Also set probabilistic weights.
        
         @parameters:        
            mass (ndarray): Length of numParticles
            temperature (float)
            q_std (float): Standard deviation in positions.
        """

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
        """
        @description:
            Return information about the particleNum th particle.
         @parameters:        
            particleNum (int): Label of particle
        """    
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[:, particleNum], self.p[:, particleNum], \
        self.mass[particleNum], self.weights[particleNum]