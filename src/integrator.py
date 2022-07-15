#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: bruno

file containing functions to compute the numerical solution of 
momenta and positions for a N-particle system

"""

import numpy as np
from ensemble import Ensemble
   
class integrator_methods:       
    """
    @description:
        This class contains functions to compute the positions 
        and momenta at final simulation time finalTime for a system of N particles 
        interacting according to a given potential.
        We do not store the solution at each time step, but the solution at final simulation
        time. 
        We assume that the particles are described by momentum and position vectors in D dimensions.
        
    @parameters:        
        samples (numpy array): train/test data points
        labels (numpy array): contains the samples/data labels
    """
    
    def __init__(self, ensemble, stepSize, finalTime):
        # initial positions 
        self.q = np.copy(ensemble.q)
        # initial momenta 
        self.p = np.copy(ensemble.p)
        self.numParticles = ensemble.numParticles
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime        
        self.totalSteps = int(self.finalTime/self.stepSize)
        # integrator to use
        self.method = None        
        self.potentialFunction = ensemble.potential
        # value of potential for a given configuration q
        self.potential = np.zeros_like(self.q)
    
    def leap_frog(self):
        """
        @description:
            function to compute numerically positions and momenta for N particles 
            using leap frog algorithm
            Algorithm taken from https://www2.math.ethz.ch/education/bachelor/seminars/fs2008/nas/crivelli.pdf
            
        @parameters:        
            ensemble (class): initilialize particles and contains information about particles
                              positons and momente
            stepSize (float): step size for numerical integrator
            finalTime (float): final simulation time
        """
        # first value of potential
        for i in range(self.numParticles):            
            self.potential[:, i] = self.potentialFunction(self.q[:, i]) 
            
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):             
            for j in range(self.numParticles):   
                # half step in leap frog
                pMidStep = self.p[:, j] - 0.50 * self.stepSize * self.potential[:, j]
                self.q[:, j]  = self.q[:, j] + self.stepSize * pMidStep
                self.p[:, j]  = pMidStep - 0.50 * self.stepSize * self.potential[:, j]  
                # update potential
                self.potential[:, j]  = self.potentialFunction(self.q[:, j])
            
        # return postion and momenta of all particles at finalTime
        return self.q, self.p
    
    def stormer_verlet(self):
        """
        @description:
            function to compute numerically positions and momenta for N particles 
            using Stormer-Verlet frog algorithm. 
            Algorithm taken from https://www2.math.ethz.ch/education/bachelor/seminars/fs2008/nas/crivelli.pdf
            
        @parameters:        
            ensemble (class): initilialize particles and contains information about particles
                              positons and momente
            stepSize (float): step size for numerical integrator
            finalTime (float): final simulation time
        """
        # first value of potential
        for i in range(self.numParticles):            
            self.potential[:, i] = self.potentialFunction(self.q[:, i]) 
        
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):
            for j in range(self.numParticles):
                self.q[:, j] = self.q[:, j] + self.stepSize * ( self.p[:, j] -  0.50 * self.stepSize * self.potential[:, j] )            
                newPotential = self.potentialFunction(self.q[:, j])
                self.p[:, j] = self.q[:, j] - 0.50 * self.stepSize * ( self.potential[:, j] + newPotential)
                self.potential[:, j] = newPotential         
        
        # return postion and momenta of all particles at finalTime
        return self.q, self.p
    
    def numerical_solution(self, method = 'LP'):
        self.method = method
        if self.method == 'LP':
            # solve trajectories for each particle
            self.leap_frog()
            # return matrix containing numerical solutions for all particles
            return self.q, self.p
        
        if self.method == 'SV':
            self.stormer_verlet()
            return self.q, self.p
        
        else:
            raise Exception('Choose a valid numerical integrator')
                        
def toy_potential(x):
    return np.sin(x) + x


