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
#from scipy.constants import Boltzmann as boltzmannConst
boltzmannConst = 1
from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
import jax
from functools import partial
import os


jax.config.update("jax_enable_x64", True)  # required or grad returns NaNs

class ReferenceIntegrator:
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

    def setStepSize(self, dt: float):
        self.stepSize = dt

    def setNumSteps(self, N: int):
        self.numSteps = N

class ReferenceLeapfrog(ReferenceIntegrator):
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
        q = np.copy(q)
        p = np.copy(p)

        v = p / mass
        # assuming gradient will yield a jax array, convert it to NumPy
        currentAccel = - np.array(self.gradient(q)) / mass


        q += 0.5 * self.stepSize*v
        for step in range(self.numSteps):
            # loop body
            currentAccel = -np.array(self.gradient(q)) / mass;
            v = v + self.stepSize * currentAccel
            q = q + self.stepSize * v
            

        p = v * mass
        # perform final position half-step on exit
        return (q + 0.5*self.stepSize * v, p)

class HMC_reference:
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
        self.integrator = ReferenceLeapfrog(stepSize, simulTime, self.gradient)

    def __hash__(self):
        """
        Needed to ensure getSamples is recompiled if any attributes change.
        """
        return hash(
            (
                self.simulTime,
                self.stepSize,
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

    def getWeight(self, q, p, mass, temperature):
        #ensure that we only work with real numbers
        H = 0.5 * np.dot(p, p) / mass + float(self.potential(q))
        return np.exp(-H / (boltzmannConst * temperature))

    def print_information(self):
        print("integrator: ", self.integrator)
        print("final integration time: ", self.simulTime)
        print("time step: ", self.stepSize)

    def propagate_ensemble(self, ensemble):
        q, p, mass, temperature, key = ensemble
        numParticles, numDimensions = q.shape
        print("[HMC] Ensemble propagation starting positions:\n", q)
        print("[HMC] Ensemble propagation starting momenta:\n", p)
        # copy and convert to NumPy arrays
        q, p, mass = (
            np.array(q),
            np.array(p),
            np.array(mass),
        )  # don't want these to be modified
        weights = np.array(mass)
        # iterate over the particles
        for i in range(numParticles):
            qi, pi, mi = q[i], p[i], mass[i]
            qi, pi, wi = self.propagate(temperature, qi, pi, mi, 1)
            q[i] = np.copy(qi)
            p[i] = np.copy(pi)
            weights[i] = np.copy(wi)

        # make new ensemble object with updated attributes and copy NumPy
        # arrays into JAX NumPy arrays
        print("[HMC] Ensemble propagation final positions:\n", q)
        print("[HMC] Ensemble propagation final momenta:\n", p)
        ensemble = Ensemble(numDimensions, numParticles, temperature, key)
        ensemble.q = jnp.array(q)
        ensemble.p = jnp.array(p)
        ensemble.weights = jnp.array(weights)

        return ensemble

    def hamiltonian(self, q, p, mass):
        return jnp.dot(p,p)/(2*mass) + self.potential(q)

    def propagate(self, temperature, q, p, mass, key):
        proposedQ, proposedP = self.integrator.integrate(q, p, mass)
        # compute the energies
        E0 = self.hamiltonian(q,p, mass)
        E1 = self.hamiltonian(proposedQ, proposedP, mass)
        alpha = np.exp( - (E1 - E0) /(boltzmannConst*temperature) )

        acceptanceProb = np.minimum(alpha, 1.0)
        print(acceptanceProb)
        u = np.random.uniform(0,1)
        #check if the move is accepted
        if u < acceptanceProb:
                q = proposedQ
                p = proposedP
#        else:
#            p *= -1;

        weight = self.getWeight(q, p, mass, temperature)

        return (q, p, weight)
