
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
from scipy.optimize import approx_fprime
from autograd import grad, jacobian
   
class integrator_methods:	
    """
    @description:
        This class contains functions to compute the positions 
        and momenta at final simulation time finalTime for a system of N particles 
        interacting according to a given potential.
        We do not store the solution at each time step, but the solution at final simulation
        time. 
        We assume that the particles are described by momentum and position vectors in D dimensions.
        Furthemore, we assume that we have the gradient of the interaction to work with and pass to
        either of the integrators
        
    @parameters:        
        ensemble : data strucuture containing all relevant information about our particles and initialize 
                   positions and momenta
        stepSize : time step to use for numerical solution
        finalTime : final simulation time
        gradientCompute: default is False since we assume we are given the gradient of the potential function
    """
    
    def __init__(self, ensemble, stepSize, finalTime, gradientCompute = False):
        # physical quantities data
        # initial positions 
        self.q = np.copy(ensemble.q)
        # initial momenta 
        self.p = np.copy(ensemble.p)

        self.acceleration = np.zeros_like(self.q)
        self.mass = ensemble.mass

        self.numParticles = ensemble.numParticles
        if False == gradientCompute:
            self.interaction = ensemble.potential
            self.computeGradient = False
        else:
            self.interaction = ensemble.potential 
            self.computeGradient = True
        self.gradientPotential = np.zeros_like(self.q)
        
        # matrix to store forces on each particle for a given configuration q
        self.force = np.zeros_like(self.q)
        
        # numerical integrator data
        # step size for integrator
        self.stepSize = stepSize
        # final simulation time
        self.finalTime = finalTime         
        # total steps in numerical solution
        self.totalSteps = int(self.finalTime/self.stepSize)
        # integrator to use
        self.method = None        

    
    # DO NOT call this function yet.
    def compute_gradient(self):
    """
		@description:
		    This function is meant to compute the gradient of the potential interaction if not given
		"""
        eps = np.sqrt(np.finfo(float).eps)
        for i in range(self.numParticles):
            self.gradientPotential[:, i] = approx_fprime(self.q[: i], self.interaction, [eps, np.sqrt(200) * eps], self.mass)
   
    
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
        if True == self.computeGradient:
            pass
            
        # first value of force
        for i in range(self.numParticles):            
            self.force[:, i] = - self.interaction(self.q[:, i]) 
   
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):             
            for j in range(self.numParticles):
                # half step in leap frog
                pMidStep = self.p[:, j] + 0.50 * self.stepSize * self.force[:, j]
                self.q[:, j]  = self.q[:, j] + self.stepSize * pMidStep
                # update force
                self.force[:, j]  = - self.interaction(self.q[:, j])
                self.p[:, j]  = pMidStep + 0.50 * self.stepSize * self.force[:, j]                  
        
        self.acceleration = self.force / self.mass
        
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
        # first value of force
        for i in range(self.numParticles):            
            self.force[:, i] = - self.interaction(self.q[:, i]) 
        
        # number of time steps consider on [initialTime, finalTime]
        for i in range(self.totalSteps):
            for j in range(self.numParticles):
                self.q[:, j] = self.q[:, j] + self.stepSize * ( self.p[:, j] +  0.50 * self.stepSize * self.force[:, j] )            
                newforce = - self.interaction(self.q[:, j])
                self.p[:, j] = self.p[:, j] + 0.50 * self.stepSize * ( self.force[:, j] + newforce)
                self.force[:, j] = newforce         
        
        self.acceleration = self.force / self.mass

    def numerical_solution(self, method = 'LP'):
        self.method = method
        if self.method == 'LP':
            # solve trajectories for each particle
            self.leap_frog()
        
        if self.method == 'SV':
            self.stormer_verlet()
        
        else:
            raise Exception('Choose a valid numerical integrator')
        
    def get_quantities(self):
        return (self.q, self.p, self.force, self.acceleration)

     
def toy_potential(x):
    return np.sin(x) + x
