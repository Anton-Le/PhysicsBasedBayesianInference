#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: bruno

file containing functions to compute the numerical solution of 
momenta and positions for a N-particle system

"""

import numpy as np
import abc

class integrator:
    def __init__(self, ensemble):
        # physical quantities data
        # initial values of p and q 
        self.q = np.copy(ensemble.q)
        # initial momenta 
        self.p = np.copy(ensemble.p)
        self.acceleration = np.zeros_like(self.q)
        self.mass = ensemble.mass
        self.numParticles = ensemble.numParticles
        
    @abc.abstractmethod
    def integrate(self, gradientPotential, stepSize, finalTime):
        raise NotImplementedError('Integrator superclass doesn\'t specify \
            integration method')
    
class leap_frog(integrator):
    def __init__(self, ensemble, gradientPotential, stepSize, finalTime):
        super().__init__(ensemble) 
        # keep potential separated from ensemble class
        # we pass the gradient of the potential to integrators
        # this is a function
        self.gradientPotential = gradientPotential
        # this is the actual value of the gradient
        self.gradientPotentialValue = np.zeros_like(self.q)
        
        # numerical integrator data
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime         
        # total steps in numerical solution
        self.totalSteps = int(self.finalTime/self.stepSize)
        
    def integrate(self):
     
        # gradient of potential at initial time
        for i in range(self.numParticles):            
            self.gradientPotentialValue[:, i] = self.gradientPotential(self.q[:, i]) 
        
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):             
            for j in range(self.numParticles):   
                # half step in leap frog                
                pMidStep = self.p[:, j] - 0.50 * self.stepSize * self.gradientPotentialValue[:, j]
                self.q[:, j]  = self.q[:, j] + self.stepSize * pMidStep
                # update force
                self.gradientPotentialValue[:, j] = self.gradientPotential(self.q[:, j])
                self.p[:, j]  = pMidStep - 0.50 * self.stepSize * self.gradientPotentialValue[:, j]  
        
        return (self.q, self.p)
    
class stormer_verlet(integrator):
    def __init__(self, ensemble, gradientPotential, stepSize, finalTime):
        super().__init__(ensemble) 
        # keep potential separated from ensemble class, this is a function
        self.gradientPotential = gradientPotential
        self.gradientPotentialValue = np.zeros_like(self.q)
        
        # numerical integrator data
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime         
        # total steps in numerical solution
        self.totalSteps = int(self.finalTime/self.stepSize)
        # integrator to use
        
    def integrate(self):
        # gradient of potential at initial time
        for i in range(self.numParticles):            
            self.gradientPotentialValue[:, i] = self.gradientPotential(self.q[:, i]) 
            
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):
            for j in range(self.numParticles):
                self.q[:, j] = self.q[:, j] + self.stepSize * ( self.p[:, j] -  0.50 * self.stepSize * self.gradientPotentialValue[:, j] )            
                newGradient = self.gradientPotential(self.q[:, j])
                self.p[:, j] = self.p[:, j] - 0.50 * self.stepSize * ( self.gradientPotentialValue[:, j] + newGradient)
                self.gradientPotentialValue[:, j] = newGradient         
        
        return (self.q, self.p)

