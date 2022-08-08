#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, pmap
from scipy.stats import norm
from scipy.constants import Boltzmann as boltzmannConst
from integrator import Leapfrog, StormerVerlet
import jax
from functools import partial
import os
os.environ['XLA_FLAGS'] ='--xla_force_host_platform_device_count=4'

jax.config.update("jax_enable_x64", True)  # required or grad returns NaNs


class HMC:
    """
    @description:
        Class with getSamples method.
    """

    def __init__(
        self,
        numDimensions,
        simulTime,
        stepSize,        
        temperature, 
        qStd, 
        density,
        potential=None,
        gradient=None,
        method="Leapfrog",
    ):
        """
        @parameters:
            simulTime (float): Duration of Hamiltonian simulation.
            stepSize (float):
            temperature (float): Determines standard deviation momentum.
            qStd (float): Intieial standard deviation of position.
            mass (float): Determines standard deviation of momentum.
            density (func): Probability density function taking position.
            potential (func): Optional function equal to -ln(density)
            gradient (func): Optional gradient of -ln(density(q))
            method (str):
        """
        # we gather all information from the ensemble
        self.numDimensions = numDimensions
        self.simulTime = simulTime
        self.stepSize = stepSize
        self.temperature = temperature
        self.qStd = qStd
        self.density = density

        if potential:
            self.potential = potential
        else:
            self.potential = self.potentialFunc

        if gradient:
            self.gradient = gradient
        else:
            self.gradient = grad(self.potential)

        if method == "Leapfrog":
            self.integrator = Leapfrog(
                stepSize, simulTime, self.gradient
            )
        elif method == "Stormer-Verlet":
            self.integrator = StormerVerlet(
                stepSize, simulTime, self.gradient
            )
        else:
            raise ValueError("Invalid integration method selected.")

    def __hash__(self): # needed for pmap
        return hash((
            self.numDimensions,
            self.simulTime,
            self.stepSize,
            self.temperature,
            self.qStd,
            self.potential,
            self.gradient,
            self.integrator,
            ))

    # negative log potential energy, depends on position q only
    # U(x) = -log( p(x) )
    def potentialFunc(self, q):
        """
        @description:
            Get potential at position q.

        @parameters:
            self.density (func):
            q (ndarray): Position
        """
        return -jnp.log(self.density(q))

    def setMomentum(self, key, mass):
        """
        @description:
            Distribute momentum based on a thermal distribution.

         @parameters:
            mass (ndarray): Length of numParticles
            temperature (float)
        """
        # thermal distribution
        pStd = jnp.sqrt(mass * boltzmannConst * self.temperature)

        return jax.random.normal(key, shape=(self.numDimensions,)) * pStd


    def getWeightRatio(self, newQ, newP, oldQ, oldP, mass):
        oldH = 0.5 * jnp.dot(oldP, oldP) / mass + self.potential(oldQ)
        newH = 0.5 * jnp.dot(newP, newP) / mass + self.potential(newQ)
        return jnp.exp(oldH - newH)

    def print_information(self):
        print("integrator: ", self.integrator)
        print("final integration time: ", self.simulTime)
        print("time step: ", self.stepSize)

    @partial(pmap, static_broadcasted_argnums=(0, 1))
    def getSamples(self, numIterations, mass, key):
        """
        @description:
            Get samples from HMC.

         @parameters:
            numIterations (int):
            temperature (float): Temperature used to set momentum.
            qStd (float): Standard deviation of initial positions.
        """

        # to store samples generated during HMC iteration.
        # This is an array of matrices, each matrix corresponds to an HMC sample
        samples = jnp.zeros((self.numDimensions, numIterations))
        momentums = jnp.zeros_like(samples)
        key, subkey = jax.random.split(key)

        q = jax.random.normal(subkey, shape=(self.numDimensions,)) * self.qStd

        initialVal = (samples, momentums, q, key)

        bodyFunc = lambda i, val: _getSamplesBody(i, val, mass, self)

        finalVal = jax.lax.fori_loop(0, numIterations, bodyFunc, initialVal)

        samples, momentums, _, _ = finalVal

        return samples, momentums

def _getSamplesBody(i, val, mass, self):
    samples, momentums, q, key = val

    key, subkey = jax.random.split(key) 

    p = self.setMomentum(subkey, mass)


    proposedQ, proposedP = self.integrator.integrate(q, p, mass)


    # flip momenta

    ratio = self.getWeightRatio(proposedQ, proposedP, q, p, mass)


    acceptanceProb = jnp.minimum(1, ratio)

    key, subkey = jax.random.split(key) 


    q, p = jnp.where(
        jax.random.uniform(subkey) < acceptanceProb,
        jnp.array([proposedQ, proposedP]),
        jnp.array([q, p]),
        )

    p = -p

    
    # update accepted moves
    samples = samples.at[:, i].set(q)
    momentums = momentums.at[:, i].set(p)

    val = (samples, momentums, q, key)

    return val

