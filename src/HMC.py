#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno
"""

import numpy as np
import jax.numpy as jnp
from jax import grad

from integrator import Leapfrog, StormerVerlet

class HMC:
    # def __init__(self, ensemble, samples, density, integrator, stepSize, finalTime, potential=None, gradient = None):
    def __init__(self, ensemble, simulTime, stepSize, density, gradient=None, method='Leapfrog'):
        # we gather all information from the ensemble
        self.ensemble = ensemble
        self.simulTime = simulTime
        self.stepSize = stepSize
        self.density = density


        if gradient:
            self.gradient = gradient
        else:
            self.gradient = grad(self.potential)


        if method == 'Leapfrog':
            self.integrator = Leapfrog(ensemble, stepSize, simulTime, gradient)
        elif method == 'Stormer-Verlet':
            self.integrator = StormerVerlet(ensemble, stepSize, simulTime, gradient)
        else:
            raise ValueError('Invalid integration method selected.')

    
    # negative log potential energy, depends on position q only
    # U(x) = -log( p(x) ) 
    def potential(self, i):    
        '''
        Get potential of i th particle
        '''
        return -jnp.log( self.density(ensemble.q[:, i]) )            
    
    def getWeights(self, q, p):   
        weights = np.zeros(self.ensemble.numParticles)
        for i in range(self.ensemble.numParticles):
            # H = kinetic_energy + potential_energy
            H = 0.5 * jnp.dot(p[:, i], p[:, i]) / self.ensemble.mass[i] + self.potential(i)
            weights[i] = jnp.exp(-H)                    
        return weights
    
    def print_information(self):
        print('integrator: ', self.integrator)
        print('final integration time: ', self.finalTime)
        print('time step: ', self.stepSize)
        print('total samples to compute: ', self.samples)
        
    def getSamples(self, numSamples, temperature, qStd):
        
        self.ensemble.initializeThermal(temperature, qStd)       
        
        # to store samples generated during HMC iteration. 
        # This is an array of matrices, each matrix corresponds to an HMC sample
        samples_hmc = np.zeros((self.ensemble.numParticles, self.ensemble.numDimensions, numSamples))
        self.print_information()
        
        for i in range(self.samples):
            if i%100 == 0:            
                print('HMC iteration ', i+1)
            

            # numerical solution for momenta and positions

            p, q = integrator.integrate()
            
            # flip momenta
            newMomentum = -newMomentum

            oldWeights = self.getWeights(self.ensemble.q, self.ensemble.p)                 
            proposedWeights = self.getWeights(q, p)    
            ratio = proposedWeights/oldWeights
            acceptanceProb = np.minimum(1, ratio)
            
            u = np.random.uniform(size=self.ensemble.numParticles)
            
            mask = u < acceptanceProb      
            self.ensemble.q[:, mask] = newPosition[mask]
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
                
            
        
