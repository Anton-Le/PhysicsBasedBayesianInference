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
import numpy as np

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
        '''
        @parameters:
            ensemble (Ensemble):
            stepSize (float):
            finalTime (float):
            gradient (func): Gradient of potential.
        '''
        # initial positions
        self.q = np.copy(ensemble.q)
        # initial momenta
        self.p = np.copy(ensemble.p)
        self.mass = np.copy(ensemble.mass)
        # calculate initial velocities
        self.v = self.p / self.mass
        self.numParticles = ensemble.numParticles
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime
        self.numSteps = int(self.finalTime/self.stepSize)

        # gradient function
        self.gradient = gradient


        # save avoid expenseive numerical differentiation of nBody potential
        if not gradient:
            print(f'Gradient={gradient} - performing nBody simulation.')
            self.getAccel = self.getAccelNBody
            

    def getAccel(self, i):
        """
        @description:
            Get acceleration of the i th particle.

        @parameters:
            self.q (ndarray): numDimensions x numParticles array
            self.mass (ndarray):
            self.potential (func):
            self.dq (float):
        """

        return  -self.gradient(self.q[:, i]) / self.mass[i]

    def getAccelNBody(self, i):
        """
        @description:
            Calculate acceleration of i th particle in N-body system.
            
        @parameters:        
            q (ndarray): numDimensions x numParticles array of positions
            i (int): index
            mass (ndarray): numParticles array of masses
        """
        return getAccelNBody(self.q, self.mass, i)
            

    def integrate(self):
        raise NotImplementedError('Integrator superclass doesn\'t specify \
            integration method')


class Leapfrog(Integrator):
    def integrate(self):
        """
        @description:
            function to compute numerically positions and momenta for N particles
            using leap frog algorithm
            Integer formulation from https://en.wikipedia.org/wiki/Leapfrog_integration

        @parameters:
        """

        for i in range(self.numParticles):
            currentAccel = self.getAccel(i)
            # number of time steps consider on [initialTime, finalTime]

            for j in range(self.numSteps):
                self.q[:, i] += self.v[:, i] * self.stepSize + 0.5 * currentAccel * self.stepSize ** 2
                nextAccel = self.getAccel(i)
                self.v[:, i] += 0.5 * (currentAccel + nextAccel) * self.stepSize
                currentAccel = np.copy(nextAccel)

            self.p[:, i] = self.v[:, i] * self.mass[i]

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
        for i in range(self.numParticles):
            qPast = np.copy(self.q[:, i])
            self.q[:, i] = self.q[:, i] + self.v[:, i] * self.stepSize + 0.5 * self.getAccel(i) * self.stepSize ** 2

            # number of time steps consider on [initialTime, finalTime]
            for j in range(self.numSteps):
                temp = np.copy(self.q[:, i])
                self.q[:, i] = 2 * self.q[:, i] - qPast + self.getAccel(i) * self.stepSize ** 2
                qPast = np.copy(temp)


            self.v[:, i] = (self.q[:, i] - qPast) / self.stepSize
            self.p[:, i] = self.v[:, i] * self.mass[i]
        # return postion and momenta of all particles at finalTime
        return (self.q, self.p)
