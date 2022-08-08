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

    def __init__(self, numParticles, numDimensions):
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
        self.q = np.zeros((numParticles, numDimensions))
        self.p = np.zeros((numParticles, numDimensions))
        self.mass = np.ones(numParticles)

    def __iter__(self):
        """
        @description:
            Allows for easy unpacking of ensemble object.
        """
        return self.q, self.p, self.mass

    # def setWeights(self, temperature):
    #     """
    #     @description:
    #         Set probabilistic weights.
    #      @parameters:
    #         temperature (float):
    #     """
    #     kineticEnergy = np.sum((self.p ** 2 / (2 * self.mass)), axis=0)
    #     hamiltonian = self.potential(self.q) + kineticEnergy
    #     self.weights = np.exp(- hamiltonian / (boltzmannConst * temperature))

    def setPosition(self, qStd):
        """
        @description:
            Distribute position with a normal distribution.
         @parameters:
            mass (ndarray): Length of numParticles
            q_std (float): Standard deviation in positions.
        """


        self.q = norm.rvs(
            scale=qStd, size=(self.numParticles, self.numDimensions)
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
        print(f'{self.p.shape=}')
        print(f'{pStd.shape=}')
        print((self.numParticles, self.numDimensions))
        self.p = norm.rvs(
            scale=pStd[:, None], size=(self.numParticles, self.numDimensions)
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
            self.q[particleNum],
            self.p[particleNum],
            self.mass[particleNum],
        )
