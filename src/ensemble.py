#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: thomas

Contains Ensemble class.

"""

import jax.numpy as jnp
import jax
from jax import vmap, jit
from scipy.stats import norm
#from scipy.constants import k as boltzmannConst
boltzmannConst = 1
from potential import nBodyPotential
from functools import partial

jax.config.update(
    "jax_enable_x64", True
)  # required or mass (1e-27) * Boltzmann is too small -> 0


class Ensemble:
    """
    @description:
        Data structure containing information of an ensemble.
        Contains positions, momenta, masses, probabilistic weights of each
        particle, as well as the potential function.
    """

    def __init__(self, numDimensions, numParticles, temperature, key):
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
        self.temperature = temperature
        self.ZPE = 0.0
        self.q = jnp.zeros((numParticles, numDimensions))
        self.p = jnp.zeros((numParticles, numDimensions))
        self.mass = jnp.ones(numParticles)
        self.initWeights = jnp.ones(numParticles) * (-jnp.log(numParticles))
        self.weights = jnp.ones(numParticles) #* 1.0/self.numParticles
        self.key = key

    def __iter__(self):
        """
        @description:
            Allows for easy unpacking of ensemble object.
        """
        things = (self.q, self.p, self.mass, self.temperature, self.key)
        return iter(things)

    def __str__(self):
        return f"Ensemble: \n q:{self.q} \n p:{self.p} \n"

    def setPosition(self, qStd=1, positionFunc=None):
        """
        @description:
            Distribute position with a normal distribution.
         @parameters:
            qStd (float): Standard deviation of initial position. Only used if 
                            positionFunc not specified.
            positionFunc (func): Init function taking jax PRNGkey, shape as 
                            only args
        """
        self.key, subkey = jax.random.split(self.key)
        shape=(self.numParticles, self.numDimensions,)

        if positionFunc:
            self.q = positionFunc(subkey, shape)

        else:
            self.q = jax.random.multivariate_normal(subkey,
                                                jnp.zeros(self.numDimensions),
                                                jnp.identity(self.numDimensions)*qStd**2,
                                                shape=(self.numParticles,) )
        return self.q

    def setMomentum(self):
        """
        @description:
            Distribute momentum based on a thermal distribution.

         @parameters:
            mass (ndarray): Length of numParticles
            temperature (float)
        """
        self.key, subkey = jax.random.split(self.key)

        pStd = jnp.sqrt( boltzmannConst * self.temperature)
        #self.p = 1/pStd**3 * jax.random.multivariate_normal(subkey,
        #                                        jnp.zeros(self.numDimensions),
        #                                        jnp.identity(self.numDimensions)*self.temperature*boltzmannConst,
        #                                        shape=(self.numParticles,) )
        self.p = jax.random.normal(
            subkey,
            shape=(
                self.numParticles,
                self.numDimensions,
            ),
        )
        return self.p

    def setWeights(self, potential):
        self.weights = self._setWeights(potential, self.q, self.p, self.mass)
        return self.weights

    def rescaleTemperature(self):
        """
        This function is intended to rescale the temperature to
        ensure that all energies are somewhat in the same ballpark.
        """
        E_max = float(jnp.min( jnp.abs( self.weights ) ) )
        self.temperature = self.temperature / E_max;
        self.weights = self.weights * (boltzmannConst * self.temperature)
        return self.temperature

    def shiftEnergy(self):
        """
        Shift energies (weights). Corresponds to a reset of the (absolute)
        energy 0 of the potential.
        """
        self.ZPE = float(jnp.max( self.weights )) #since we store negative energies
        self.weights -= self.ZPE
        return self.ZPE

    def getWeightedMean(self):
        return jnp.average(self.q, axis=0, weights=jnp.exp(self.weights), returned=True)
    
    def getArithmeticMean(self):
        return jnp.average(self.q, axis=0) 

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
            self.weights[particleNum],
            self.initWeights[particleNum],
        )

    @partial(vmap, in_axes=(None, None, 0, 0, 0))
    def _setWeights(self, potential, q, p, mass):
        H = 0.5 * jnp.dot(p, p) / mass + potential(q)
        return -H / (boltzmannConst * self.temperature)
