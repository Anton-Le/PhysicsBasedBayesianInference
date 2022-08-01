#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: bruno
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.stats import norm
from scipy.constants import Boltzmann as boltzmannConst
from integrator import Leapfrog, StormerVerlet
import jax


class HMC:
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
            self.integrator = Leapfrog(ensemble, stepSize, simulTime, self.gradient)
        elif method == 'Stormer-Verlet':
            self.integrator = StormerVerlet(ensemble, stepSize, simulTime, self.gradient)
        else:
            raise ValueError('Invalid integration method selected.')

    
    # negative log potential energy, depends on position q only
    # U(x) = -log( p(x) ) 
    def potential(self, q):
        '''
        Get potential at position q.
        '''
        return -jnp.log( self.density(q) )       


    def getWeights(self, q, p):   
        weights = np.zeros(self.ensemble.numParticles)
        for i in range(self.ensemble.numParticles):
            # H = kinetic_energy + potential_energy
            H = 0.5 * jnp.dot(p[:, i], p[:, i]) / self.ensemble.mass[i] + self.potential(q[:, i])
            weights[i] = jnp.exp(-H)                    
        return weights
    
    def print_information(self):
        print('integrator: ', self.integrator)
        print('final integration time: ', self.simulTime)
        print('time step: ', self.stepSize)

        

    def getSamples(self, numSamples, temperature, qStd):
        
            
        
        # to store samples generated during HMC iteration. 
        # This is an array of matrices, each matrix corresponds to an HMC sample
        samples_hmc = np.zeros((self.ensemble.numDimensions, self.ensemble.numParticles, numSamples))
        shape = samples_hmc.shape
        momentum_hmc = np.zeros_like(samples_hmc)
        self.print_information()
        self.integrator.q = self.ensemble.setQ(qStd)

        
        for i in range(numSamples):
            if i%100 == 0:            
                print('HMC iteration ', i+1)

            self.integrator.p = self.ensemble.setP(temperature)
            oldQ = np.copy(self.integrator.q)
            oldP = np.copy(self.integrator.p)
            oldWeights = self.getWeights(self.integrator.q, self.integrator.p)    

            # numerical solution for momenta and positions

            q, p = self.integrator.integrate()
            
            # flip momenta
            p = -p

                            
            proposedWeights = self.getWeights(q, p)    
            ratio = proposedWeights/oldWeights


            acceptanceProb = np.minimum(1, ratio)


            u = np.random.uniform(size=self.ensemble.numParticles)

            
            # keep updated position/momenta unless:
            mask = u > acceptanceProb


      
            self.integrator.q[:, mask] = oldQ[:, mask]
            self.integrator.p[:, mask] = oldQ[:, mask]
            # update accepted moves
            samples_hmc[:, :, i] = self.integrator.q
            momentum_hmc[:, :, i] = self.integrator.p

            # Is it a problem that we add the same point to samples twice if a proposal is rejected? I am not sure
            
        return samples_hmc, momentum_hmc
    
# toy examples to test HMC implementation on a 2D std Gaussian distribution
def density(x):
    return jnp.exp( -0.50*jnp.linalg.norm(x)**2 ) / jnp.sqrt((2*jnp.pi)**2) 
# this is passed to the integrator
def potential(x):    
    return -jnp.log( density(x) )
                
            
        
