#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: thomas

Contains Ensemble class.

"""

import numpy as np
from scipy.stats import norm
from scipy.constants import k as boltzmannConst
from potential import nBodyPotential


class Ensemble:
    """
    @description:
        Data structure containing information of an ensemble.
        Contains positions, momenta, masses, probabilistic weights of each
        particle, as well as the potential function.
    """

    def __init__(self, numDimensions, numParticles):
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
        self.mass = np.ones(numParticles)
        self.weights = np.zeros(numParticles)

    def __iter__(self):
        """
        @description:
            Allows for easy unpacking of ensemble object.
        """
        return self.q, self.p, self.mass, self.weights, self.potential

    def setWeights(self, potential, temperature):
         """
         @description:
             Set probabilistic weights.
          @parameters:
             potential (function):
         """
         for particleId in range(self.numParticles):
             kineticEnergy = np.sum((self.p[:,particleId] ** 2 / (2 * self.mass[particleId])), axis=0)
             hamiltonian = potential(self.q[:,particleId]) + kineticEnergy
             self.weights[particleId] = np.exp(- hamiltonian / (boltzmannConst * temperature))

    def setPosition(self, qStd):
        """
        @description:
            Distribute position with a normal distribution.
         @parameters:
            mass (ndarray): Length of numParticles
            q_std (float): Standard deviation in positions.
        """

        self.q = norm.rvs(
            scale=qStd, size=(self.numDimensions, self.numParticles)
        )

        return self.q

    def setMomentum(self, temperature):
        """
        @description:
            Distribute momentum based on a thermal distribution.

         @parameters:
            mass (ndarray): Length of numParticles
            temperature (float)
        """
        # thermal distribution
        pStd = np.sqrt(self.mass * boltzmannConst * temperature)
        self.p = norm.rvs(
            scale=pStd, size=(self.numDimensions, self.numParticles)
        )

        return self.p

    def particle(self, particleNum):
        """
        @description:
            Return information about the particleNum th particle.
         @parameters:
            particleNum (int): Label of particle
        """
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(
                f"Index {particleNum} out of bounds. "
                f"numParticles={self.numParticles}"
            )

        return (
            self.q[:, particleNum],
            self.p[:, particleNum],
            self.mass[particleNum],
            self.weights[particleNum],
        )
