#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 12:44:20 2022

@author: bruno
"""
import sys
# setting path
sys.path.append('../')
  
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

from ensemble import Ensemble
from HMC import HMC


# this is the density we want to sample from 
def density(x):
    return jnp.exp( -0.50*jnp.linalg.norm(x)**2 ) / jnp.sqrt((2*jnp.pi)**2) 

# potential equals -log(density)
def potential(x):    
    return -jnp.log( density(x) )
                
########################
def main():
    # ensemble setup
    numDimensions = 2
    numParticles = 1
    mass = np.ones(numParticles)
    temperature = 1
    q_std = 0.1
    
    # integrator setup
    finalTime = 2
    stepSize = 0.05
    
    # HMC setup 
    samples = 1000
    density_function = density
    
    # generate and initialize ensemble
    ensemble1 = Ensemble(numDimensions, numParticles, potential)
    ensemble1.initializeThermal(mass, temperature, q_std)
    
    # HMC algorithm
    hmc_result = HMC(ensemble1, samples, density_function, 'sv', stepSize, finalTime  )
    hmc_samples = hmc_result.hmc_sample()
    
    # we test implementation on a 2D standard normal distribution
    ## theoretical resulst for 2D standard Gaussian distribution
    mean = np.zeros(numDimensions)
    cov = np.identity(numDimensions)
    normal = np.zeros(numParticles*numDimensions*samples).reshape((numDimensions,numParticles, -1))
    for k in range(samples):
        normal[:, : , k] = np.random.multivariate_normal(mean, cov, numParticles).T
    
    # plot results normal distribution
    fig, ax = plt.subplots()
    ax.plot( hmc_samples[0, 0, :], hmc_samples[1, 0, :], label = "HMC", marker = '*', c='k', lw =0.2, ls ='-', markersize=4)
    ax.plot( normal[0, 0, :], normal[1, 0, :], label = "theoretical", marker = '.', c='r', lw =0.2, ls ='-', markersize=4)
    plt.title(r'2D standard Gaussian')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.legend(loc='upper right')
    plt.show()
            
if __name__ == '__main__':
    hmc_samples = main()
    
    
