#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno, thomas
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, vmap, jit
from scipy.stats import norm
from scipy.constants import Boltzmann as boltzmannConst
from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
import jax
from functools import partial
import os
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

jax.config.update("jax_enable_x64", True)  # required or grad returns NaNs


class HMC:
    """
    @description:
        Class with getSamples method.
    """

    def __init__(
        self,
        # numDimensions,
        simulTime,
        stepSize,
        # temperature,
        # qStd,
        density,
        potential=None,
        gradient=None,
        method="Leapfrog",
    ):
        """
        @parameters:

            # numDimensions (int):
            simulTime (float): Duration of Hamiltonian simulation.
            stepSize (float):
            # temperature (float): Determines standard deviation of momentum.
            # qStd (float): Initial standard deviation of position.
            density (func): Probability density function taking position as only arg.
            potential (func): Optional function equal to -ln(density)
            gradient (func): Optional gradient of potential
            method (str):
        """
        # self.numDimensions = numDimensions
        self.simulTime = simulTime
        self.stepSize = stepSize
        # self.temperature = temperature
        # self.qStd = qStd
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
            self.integrator = Leapfrog(stepSize, simulTime, self.gradient)
        elif method == "Stormer-Verlet":
            self.integrator = StormerVerlet(stepSize, simulTime, self.gradient)
        else:
            raise ValueError("Invalid integration method selected.")

    def __hash__(self):
        """
        Needed to ensure getSamples is recompiled if any attributes change.
        """
        return hash(
            (
                # self.numDimensions,
                self.simulTime,
                self.stepSize,
                # self.temperature,
                # self.qStd,
                self.potential,
                self.gradient,
                self.integrator,
            )
        )

    def potentialFunc(self, q):
        """
        @description:
            Default potential function at position q.

        @parameters:
            self.density (func):
            q (ndarray): Position
        """
        return -jnp.log(self.density(q))

    # def setMomentum(self, key, mass):
    #     """
    #     @description:
    #         Distribute momentum based on a thermal distribution.

    #      @parameters:
    #         key (PRNGKeyArray)
    #         mass (ndarray): Has length numParticles
    #     """
    #     # thermal distribution
    #     pStd = jnp.sqrt(mass * boltzmannConst * self.temperature)

    #     return jax.random.normal(key, shape=(self.numDimensions,)) * pStd

    def getWeightRatio(self, newQ, newP, oldQ, oldP, mass, temperature):
        """
        @description:
            Get ratio of exp(H) for HMC

         @parameters:
            newQ (ndarray)
            newP (ndarray)
            oldQ (ndarray)
            oldP (ndarray)
            mass (float)
        """
        #oldH = 0.5 * jnp.dot(oldP, oldP) / mass + self.potential(oldQ)
        old = self.getWeight(oldQ, oldP, mass, temperature)
        new = self.getWeight(newQ, newP, mass, temperature)
        #newH = 0.5 * jnp.dot(newP, newP) / mass + self.potential(newQ)
        return jnp.exp( -(old - new) ) #(oldH - newH) / (boltzmannConst * temperature))

    def getWeight(self, q, p, mass, temperature):
        H = 0.5 * jnp.dot(p, p) / mass + self.potential(q)
        return -H / (boltzmannConst * temperature)#)

    def print_information(self):
        print("integrator: ", self.integrator)
        print("final integration time: ", self.simulTime)
        print("time step: ", self.stepSize)

    def propagate_ensemble(self, ensemble):
        q, p, mass, temperature, key = ensemble
        numParticles, _ = q.shape

        key, *keys = jax.random.split(
            key, numParticles + 1
        )  # key is returned to be used again
        keys = jnp.array(keys)

        #q, p, mass = jnp.copy(q), jnp.copy(p), jnp.copy(mass)

        q, p, weights = self.propgate(temperature, q, p, mass, keys)

        # make new ensemble object with updated attributes

        #ensemble = Ensemble(numDimensions, numParticles, temperature, key)
        ensemble.q = jnp.copy(q)
        ensemble.p = jnp.copy(p)
        ensemble.weights = jnp.copy(weights)
        ensemble.key = key

        return ensemble

    @partial(
        jit,
        static_argnums=(
            0,
            1,
        ),
    )
    @partial(vmap, in_axes=[None, None, 0, 0, 0, 0])
    def propgate(self, temperature, q, p, mass, key):
        proposedQ, proposedP = self.integrator.integrate(q, p, mass)

        weightRatio = self.getWeightRatio(
            proposedQ,
            proposedP,
            q,
            p,
            mass,
            temperature
            )

        acceptanceProb = jnp.minimum(weightRatio, 1)
        q, p = jnp.where(
            jax.random.uniform(key) < acceptanceProb,
            jnp.array([proposedQ, proposedP]),
            jnp.array([q, p]),
        )

        weight = self.getWeight(q, p, mass, temperature)

        return (q, p, weight)
