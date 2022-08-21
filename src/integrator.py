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
from jax import pmap, vmap, jit
from functools import partial
import os
# os.environ['XLA_FLAGS'] ='--xla_force_host_platform_device_count=4'


class Integrator:
    """
    @description:
        Simulate a particle's motion given its mass, initial position and 
        momentum as well as the gradient of the potential it is in.
    """
    def __init__(self, stepSize, finalTime, gradient):
        '''
        @parameters:
            stepSize (float):
            finalTime (float):
            gradient (func): Gradient of potential taking q as only argument.
        '''
        self.stepSize = float(stepSize)
        self.finalTime = finalTime

        self.numSteps = int(self.finalTime/self.stepSize)

        self.gradient = gradient

    def __hash__(self):
        return hash((
            self.stepSize,
            self.finalTime,
            self.gradient,
            self.integrate
            ))
    
    def __eq__(self, other):
        return (isinstance(other, Integrator) and 
            (self.stepSize, self.finalTime, self.gradient, self.integrate) == 
            (other.stepSize, other.finalTime, other.gradient, other.integrate))

            
    def integrate(self):
        raise NotImplementedError('Integrator superclass doesn\'t specify \
            integration method')


    # @partial(jit, static_argnums=0)
    @partial(vmap, in_axes=(None, 0, 0, 0))
    def pintegrate(self, q, p, mass):
        return self.integrate(q, p, mass)

    # @partial(jit, static_argnums=0)
    # def pintegrate(self, q, p, mass):
    #     q_p_mass = (q, p, mass)

    #     f = lambda q_p_mass: self.integrate(*q_p_mass)
    #     q, p = jax.lax.map(f, q_p_mass)
    #     return (q, p)


class Leapfrog(Integrator):
    def integrate(self, q, p, mass):
        """
        @description:
            Leap frog algorithm.
            Integer formulation from https://en.wikipedia.org/wiki/Leapfrog_integration

        @parameters:
            q (ndarray) : Initial position
            p (ndarray) : Initial momentum
            mass (float) :
        """
        q = jnp.copy(q)
        p = jnp.copy(p)

        v = p / mass           
        currentAccel = - self.gradient(q) / mass


        initial_val = (q, v, currentAccel)

        body_func = lambda i, val: _leapfrogBodyFunc(i, val, self.stepSize, self.gradient, mass)

        final_val = jax.lax.fori_loop(0, self.numSteps, body_func, initial_val)


        q, v, _ = final_val
        p = v * mass

        return (q, p)



class StormerVerlet(Integrator):
    def integrate(self, q, p, mass):
        """
        @description:
            Stormer-Verlet algorithm.
            Algorithm taken from https://www2.math.ethz.ch/education/bachelor/seminars/fs2008/nas/crivelli.pdf

        @parameters:
            q (ndarray) : Initial position
            p (ndarray) : Initial momentum
            mass (float) :
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