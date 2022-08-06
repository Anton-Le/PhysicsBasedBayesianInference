#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022

@author: bruno, thomas

file containing functions to compute the numerical solution of
momenta and positions for a N-particle system

"""

from ensemble import Ensemble
from scipy.optimize import approx_fprime
from potential import getAccelNBody, nBodyPotential
from scipy.constants import G  # for debug
import numpy as np
from functools import partial
from jax import pmap
import jax.numpy as jnp
import jax

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4' # for 4-core CPU


class Integrator:
    """
    @description:
        This class contains functions to compute the positions
        and momenta at final simulation time finalTime for a system of N particles
        interacting according to a given potential.
        We do not store the solution at each time step, but the solution at final simulation
        time.
    """

    def __init__(self, ensemble, stepSize, finalTime, gradient):
        """
        @parameters:
            ensemble (Ensemble):
            stepSize (float):
            finalTime (float):
            gradient (func): Gradient of potential.
        """
        # initial positions
        self.ensemble = ensemble
        self.q = ensemble.q
        # initial momenta
        self.p = ensemble.p
        self.mass = ensemble.mass[None, :]
        # calculate initial velocities
        self.v = self.p / self.mass
        self.numParticles = ensemble.numParticles
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime
        self.numSteps = int(self.finalTime / self.stepSize)

        # gradient function
        self.gradient = gradient

        # save avoid expenseive numerical differentiation of nBody potential
    #     if not gradient:
    #         print(f"Gradient={gradient} - performing nBody simulation.")
    #         self.getAccel = self.getAccelNBody

    # def getAccel(self, q, mass):
    #     """
    #     @description:
    #         Get acceleration of the i th particle.

    #     @parameters:
    #         self.q (ndarray): numDimensions x numParticles array
    #         self.mass (ndarray):
    #         self.potential (func):
    #         self.dq (float):
    #     """

    #     return -self.gradient(self.q[:, i]) / self.mass[i]

    # def getAccelNBody(self, i):
    #     """
    #     @description:
    #         Calculate acceleration of i th particle in N-body system.

    #     @parameters:
    #         q (ndarray): numDimensions x numParticles array of positions
    #         i (int): index
    #         mass (ndarray): numParticles array of masses
    #     """
    #     return getAccelNBody(self.q, self.mass, i)

    # def integrate(self):
    #     raise NotImplementedError(
    #         "Integrator superclass doesn't specify \
    #         integration method"
    #     )


class Leapfrog(Integrator):
    def integrate(self):
        """
        @description:
            function to compute numerically positions and momenta for N particles
            using leap frog algorithm
            Integer formulation from https://en.wikipedia.org/wiki/Leapfrog_integration

        @parameters:
        """

        self.q, self.p = _leapfrog(self.stepSize, self.numSteps, self.gradient, self.q, self.p, self.mass)

        # return postion and momenta of all particles at finalTime
        return (self.q, self.p)


class StormerVerlet(Integrator):
    def integrate(self):
        """
        @description:
            function to compute numerically positions and momenta for N particles
            using Stormer-Verlet algorithm.
            Algorithm taken from https://www2.math.ethz.ch/education/bachelor/seminars/fs2008/nas/crivelli.pdf

        @parameters:
            ensemble (class): initilialize particles and contains information about particles
                              positons and momente
            stepSize (float): step size for numerical integrator
            finalTime (float): final simulation time
        """

        # for each particle
        self.q, self.p = _stormerVerlet(self.stepSize, self.numSteps, self.gradient, self.q, self.p, self.mass)
        return (self.q, self.p)


# Helper Functions:

@partial(pmap, static_broadcasted_argnums=(0, 1, 2), in_axes=1, out_axes=1)
def _leapfrog(stepSize, numSteps, gradient, q, p, mass):
    v = p / mass
            
    currentAccel = - gradient(q) / mass
    # number of time steps consider on [initialTime, finalTime]

    initial_val = (q, v, currentAccel)

    body_func = lambda i, val: _leapfrogBodyFunc(i, val, stepSize, gradient, mass)

    final_val = jax.lax.fori_loop(0, numSteps, body_func, initial_val)

    q, v, _ = final_val

    p = v * mass
    return (q, p)


def _leapfrogBodyFunc(i, val, stepSize, gradient, mass):
    q, v, currentAccel = val
    q = q + v * stepSize + 0.5 * currentAccel * stepSize ** 2
    nextAccel = - gradient(q) / mass
    v = v + 0.5 * (currentAccel + nextAccel) * stepSize
    currentAccel = jnp.copy(nextAccel)
    val = (q, v, currentAccel)
    
    return val


@partial(pmap, static_broadcasted_argnums=(0, 1, 2), in_axes=1, out_axes=1)
def _stormerVerlet(stepSize, numSteps, gradient, q, p, mass):
    v = p / mass

    qPast = jnp.copy(q)
    q = q + v * stepSize - 0.5 * stepSize ** 2 * gradient(q) / mass

    initial_val = (q, qPast)
    
    body_func = lambda i, val: _stormerVerletBodyFunc(i, val, stepSize, gradient, mass)

    final_val = jax.lax.fori_loop(0, numSteps, body_func, initial_val)

    q, qPast = final_val
    v = (q - qPast) / stepSize
    p = v * mass


    return (q, p)


def _stormerVerletBodyFunc(i, val, stepSize, gradient, mass):
    q, qPast = val
    temp = jnp.copy(q)
    q = 2 * q - qPast - stepSize ** 2 * gradient(q) / mass
    qPast = jnp.copy(temp)
    val = (q, qPast)
    return val