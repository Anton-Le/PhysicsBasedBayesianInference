#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.stats import norm
from scipy.constants import Boltzmann as boltzmannConst
from integrator import Leapfrog, StormerVerlet
import jax

jax.config.update("jax_enable_x64", True)  # required or grad returns NaNs


class HMC:
    """
    @description:
        Class with getSamples method.
    """

    def __init__(
        self,
        ensemble,
        simulTime,
        stepSize,
        density,
        potential=None,
        gradient=None,
        method="Leapfrog",
    ):
        """
        @parameters:
            ensemble (Ensemble):
            simulTime (float): Duration of Hamiltonian simulation.
            stepSize (float):
            density (func): Probability density function taking position.
            potential (func): Optional function equal to -ln(density)
            gradient (func): Optional gradient of -ln(density(q))
            method (str):
        """
        # we gather all information from the ensemble
        self.ensemble = ensemble
        self.simulTime = simulTime
        self.stepSize = stepSize
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
                ensemble, stepSize, simulTime, self.gradient
            )
        elif method == "Stormer-Verlet":
            self.integrator = StormerVerlet(
                ensemble, stepSize, simulTime, self.gradient
            )
        else:
            raise ValueError("Invalid integration method selected.")

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

    def getWeights(self, q, p):
        """
        @description:
            Get probabilistic weights of proposed position/momentum step.

        @parameters:
            q (ndarray): numDimensions x numParticles array
            p (ndarray): numDimensions x numParticles array
            self.mass (ndarray):
            self.potential (func): Taking q[:, i]
        """
        weights = jnp.zeros(self.ensemble.numParticles)
        for i in range(self.ensemble.numParticles):
            # H = kinetic_energy + potential_energy
            H = 0.5 * jnp.dot(p[:, i], p[:, i]) / self.ensemble.mass[
                i
            ] + self.potential(q[:, i])
            weights.at[i].set( jnp.exp(-H) )
        return weights

    def getWeightsRatio(self, newQ, newP, oldQ, oldP):
        weightsRatio = np.zeros(self.ensemble.numParticles)
        for i in range(self.ensemble.numParticles):
            oldH = 0.5 * jnp.dot(oldP[:, i], oldP[:, i]) / self.ensemble.mass[
                i
            ] + self.potential(oldQ[:, i])
            newH = 0.5 * jnp.dot(newP[:, i], newP[:, i]) / self.ensemble.mass[
                i
            ] + self.potential(newQ[:, i])
            weightsRatio[i] = jnp.exp(oldH - newH)
        return weightsRatio

    def print_information(self):
        print("integrator: ", self.integrator)
        print("final integration time: ", self.simulTime)
        print("time step: ", self.stepSize)

    def getSamples(self, numSamples, temperature, qStd):
        """
        @description:
            Get samples from HMC.

         @parameters:
            numSamples (int):
            temperature (float): Temperature used to set momentum.
            qStd (float): Standard deviation of initial positions.
        """

        # to store samples generated during HMC iteration.
        # This is an array of matrices, each matrix corresponds to an HMC sample
        samples_hmc = jnp.zeros(
            (
                self.ensemble.numDimensions,
                self.ensemble.numParticles,
                numSamples,
            )
        )
        shape = samples_hmc.shape

        momentum_hmc = jnp.zeros_like(samples_hmc)

        self.print_information()
        self.integrator.q = self.ensemble.setPosition(qStd)

        for i in range(numSamples):
            if i % 100 == 0:
                print("HMC iteration ", i + 1)

            self.integrator.p = self.ensemble.setMomentum(temperature)

            oldQ = jnp.copy(self.integrator.q)
            oldP = jnp.copy(self.integrator.p)

            # numerical solution for momenta and positions

            q, p = self.integrator.integrate()

            # flip momenta
            p = -p

            ratio = self.getWeightsRatio(q, p, oldQ, oldP)

            acceptanceProb = jnp.minimum(1, ratio)

            u = np.random.uniform(size=self.ensemble.numParticles)

            # keep updated position/momenta unless:
            mask = u > acceptanceProb

            self.integrator.q.at[:, mask].set(oldQ[:, mask])
            self.integrator.p.at[:, mask].set(oldQ[:, mask])
            # update accepted moves
            samples_hmc = samples_hmc.at[:, :, i].set(self.integrator.q)
            momentum_hmc = momentum_hmc.at[:, :, i].set(self.integrator.p) 

            # Is it a problem that we add the same point to samples twice if a proposal is rejected? I am not sure

        return samples_hmc, momentum_hmc


# toy examples to test HMC implementation on a 2D std Gaussian distribution
def density(x):
    return jnp.exp(-0.50 * jnp.linalg.norm(x) ** 2) / jnp.sqrt(
        (2 * jnp.pi) ** 2
    )


# this is passed to the integrator
def potential(x):
    return -jnp.log(density(x))
