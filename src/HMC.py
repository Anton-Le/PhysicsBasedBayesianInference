#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno
"""

import numpy as np
import jax.numpy as jnp
from jax import grad

from integrator import leap_frog, stormer_verlet

class HMC:
    def __init__(self, ensemble, samples, density, integrator, stepSize, finalTime, potential=None, gradient = None):
        # we gather all information from the ensemble
        self.ensemble = ensemble
        self.samples = samples
        
        # setting potential interaction
        # if we are given the potential
        if potential:
            self.potential = potential
        # if we are given the density we want to sample from 
        else:
            self.density = density
            self.potential = self.potential_energy            
        # if we are given the gradient of the potential, i.e., the force field
        if gradient:
            self.gradient = gradient
        else:
            self.gradient = None        

        self.stepSize= stepSize
        self.finalTime = finalTime
        self.integrator = integrator
    
    # negative log potential energy, depends on position q only
    # U(x) = -log( p(x) ) 
    def potential_energy(self, x):    
        # if we are given the density we want to sample from
        return -jnp.log( self.density(x) )            
    
    def exponential_hamiltonian(self, q, p):   
        energy = np.zeros(self.ensemble.numParticles)
        for i in range(self.ensemble.numParticles):
            # H = kinetic_energy + potential_energy
            H = 0.5*jnp.linalg.norm(self.ensemble.p[:, i])**2 + self.potential(self.ensemble.q[:, i])
            energy[i] = jnp.exp(-H)                    
        return energy
    
    def print_information(self):
        print('integrator: ', self.integrator)
        print('final integration time: ', self.finalTime)
        print('time step: ', self.stepSize)
        print('total samples to compute: ', self.samples)
        
    def hmc_sample(self):
        
        # mean and variance for momenta, these are fixed. Standard multidimensional Gaussian 
        mean = np.zeros(self.ensemble.numDimensions)
        cov = np.identity(self.ensemble.numDimensions)        
        
        # to store samples generated during HMC iteration. 
        # This is an array of matrices, each matrix corresponds to an HMC sample
        samples_hmc = np.zeros(self.ensemble.numParticles*self.ensemble.numDimensions*self.samples).reshape((self.ensemble.numDimensions,self.ensemble.numParticles, -1))
        
        self.print_information()
        
        for i in range(self.samples):
            if i%100 == 0:            
                print('HMC iteration ', i+1)
                
            # we modify the momenta in the ensemble
            self.ensemble.p = np.random.multivariate_normal(mean, cov, self.ensemble.numParticles).T
            
            # solve numerically for positions and momenta
            # notice that we pass the gradient of the potential function 
            if 'lp' == self.integrator:
                if self.gradient:
                    num_sol = leap_frog(self.ensemble, self.gradient, self.stepSize, self.finalTime) 
                else:
                    # we approximate the gradient of the potential
                    num_sol = leap_frog(self.ensemble, grad(self.potential), self.stepSize, self.finalTime)       
                                
            if 'sv' == self.integrator:
                if self.gradient:                                           
                    num_sol = stormer_verlet(self.ensemble, self.gradient, self.stepSize, self.finalTime)
                else:
                    # we approximate the gradient of the potential
                    num_sol = stormer_verlet(self.ensemble, grad(self.potential), self.stepSize, self.finalTime)
            
            # numerical solution for momenta and positions
            newPosition, newMomentum = num_sol.integrate()
            
            # flip momenta
            newMomentum = -newMomentum

            proposedPosition = self.exponential_hamiltonian(newPosition, newMomentum)                    
            oldPosition = self.exponential_hamiltonian(self.ensemble.q, self.ensemble.p)    
            ratio = proposedPosition/oldPosition
            acceptanceProb = np.minimum(1, ratio)
            
            u = (np.random.rand(self.ensemble.numParticles))
            
            mask = u < acceptanceProb
            mask = np.tile(mask, (self.ensemble.numDimensions, 1))            
            self.ensemble.q[mask] = newPosition[mask]
            # update accepted moves
            samples_hmc[:, :, i] = self.ensemble.q
            #print('------------')
            
        return samples_hmc
    
# toy examples to test HMC implementation on a 2D std Gaussian distribution
def density(x):
    return jnp.exp( -0.50*jnp.linalg.norm(x)**2 ) / jnp.sqrt((2*jnp.pi)**2) 
# this is passed to the integrator
def potential(x):    
    return -jnp.log( density(x) )
                
            
        
