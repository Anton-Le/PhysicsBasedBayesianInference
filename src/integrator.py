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
from scipy.constants import G # for debug
import jax.numpy as jnp
import numpy as np
import jax
from jax import pmap
from functools import partial
import os
os.environ['XLA_FLAGS'] ='--xla_force_host_platform_device_count=4'

class Integrator:
    """
    @description:
        This class contains functions to compute the positions
        and momenta at final simulation time finalTime for a system of N particles
        interacting according to a given potential.
        We do not store the solution at each time step, but the solution at final simulation
        time.
    """
   
    def __init__(self, stepSize, finalTime, gradient):
        '''
        @parameters:
            ensemble (Ensemble):
            stepSize (float):
            finalTime (float):
            gradient (func): Gradient of potential.

        '''
        # step size for integrator
        self.stepSize = float(stepSize)
        # final simulation time
        self.finalTime = finalTime
        self.numSteps = int(self.finalTime/self.stepSize)

        # gradient function
        self.gradient = gradient

    def __hash__(self):
        return hash((self.stepSize, self.finalTime, self.gradient, self.integrate))
    
    def __eq__(self, other):

        return (isinstance(other, Integrator) and 
            (self.stepSize, self.finalTime, self.gradient, self.integrate) == 
            (other.stepSize, other.finalTime, other.gradient, other.integrate))

            
    def integrate(self):
        raise NotImplementedError('Integrator superclass doesn\'t specify \
            integration method')



    @partial(pmap, static_broadcasted_argnums=0)
    def pintegrate(self, q, p, mass):
        return self.integrate(q, p, mass)


class Leapfrog(Integrator):
    def integrate(self, q, p, mass):

        """
        @description:
            function to compute numerically positions and momenta for N particles
            using leap frog algorithm
            Integer formulation from https://en.wikipedia.org/wiki/Leapfrog_integration

        @parameters:
        """
        q = jnp.copy(q)
        p = jnp.copy(p)

        v = p / mass
            
        currentAccel = - self.gradient(q) / mass
        # number of time steps consider on [initialTime, finalTime]


        initial_val = (q, v, currentAccel)

        body_func = lambda i, val: _leapfrogBodyFunc(i, val, self.stepSize, self.gradient, mass)

        final_val = jax.lax.fori_loop(0, self.numSteps, body_func, initial_val)

        q, v, _ = final_val
        
        p = v * mass

        # return postion and momenta of all particles at finalTime
        return (q, p)


class StormerVerlet(Integrator):
    def integrate(self, q, p, mass):
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
        q = jnp.copy(q)
        p = jnp.copy(p)

        v = p / mass

        qPast = jnp.copy(q)
        q = q + v * self.stepSize - 0.5 * self.stepSize ** 2 * self.gradient(q) / mass

        initial_val = (q, qPast)
        
        body_func = lambda i, val: _stormerVerletBodyFunc(i, val, self.stepSize, self.gradient, mass)

        final_val = jax.lax.fori_loop(0, self.numSteps, body_func, initial_val)


        q, qPast = final_val

        v = (q - qPast) / self.stepSize
        p = v * mass
        # return postion and momenta of all particles at finalTime
        return (q, p)


def _leapfrogBodyFunc(i, val, stepSize, gradient, mass):
    q, v, currentAccel = val
    q = q + v * stepSize + 0.5 * currentAccel * stepSize ** 2
    nextAccel = - gradient(q) / mass
    v = v + 0.5 * (currentAccel + nextAccel) * stepSize
    currentAccel = jnp.copy(nextAccel)
    val = (q, v, currentAccel)
    return val

def _stormerVerletBodyFunc(i, val, stepSize, gradient, mass):
    q, qPast = val
    temp = jnp.copy(q)
    q = 2 * q - qPast - stepSize ** 2 * gradient(q) / mass
    qPast = jnp.copy(temp)
    val = (q, qPast)
    return val

