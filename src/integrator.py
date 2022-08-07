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
<<<<<<< HEAD
from scipy.constants import G  # for debug
import numpy as np
=======
from scipy.constants import G # for debug
import jax.numpy as jnp
>>>>>>> origin/feature-jax


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
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime
        self.numSteps = int(self.finalTime / self.stepSize)

        # gradient function
        self.gradient = gradient

        # save avoid expenseive numerical differentiation of nBody potential
        # if not gradient:
        #     print(f'Gradient={gradient} - performing nBody simulation.')
        #     self.getAccel = self.getAccelNBody


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

    #     return  -self.gradient(q) / mass

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
            

    def integrate(self):
        raise NotImplementedError(
            "Integrator superclass doesn't specify \
            integration method"
        )


class Leapfrog(Integrator):
    def integrate(self, p, q, mass):
        """
        @description:
            function to compute numerically positions and momenta for N particles
            using leap frog algorithm
            Integer formulation from https://en.wikipedia.org/wiki/Leapfrog_integration

        @parameters:
        """
        
        v = p / mass
            
        currentAccel = - self.gradient(q) / mass
        # number of time steps consider on [initialTime, finalTime]

        for j in range(self.numSteps):
            q = q + v * self.stepSize + 0.5 * currentAccel * self.stepSize ** 2
            nextAccel = - self.gradient(q) / mass
            v = v + 0.5 * (currentAccel + nextAccel) * self.stepSize
            currentAccel = jnp.copy(nextAccel)

        p = v * mass

        # return postion and momenta of all particles at finalTime
        return (q, p)


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

        v = p / mass


        qPast = jnp.copy(q)
        q = q + v * self.stepSize - 0.5 * self.stepSize ** 2 * self.gradient(q) / mass

        # number of time steps consider on [initialTime, finalTime]
        for j in range(self.numSteps):
            temp = jnp.copy(q)
            q = 2 * q - qPast - self.stepSize ** 2 * self.gradient(q) / mass
            qPast = jnp.copy(temp)


        v = (q - qPast) / self.stepSize
        p = v * mass
        # return postion and momenta of all particles at finalTime
        return (q, p)
