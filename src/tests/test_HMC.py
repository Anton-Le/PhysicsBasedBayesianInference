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
from scipy.constants import Boltzmann
from jax.scipy.stats import multivariate_normal
import jax
from ensemble import Ensemble
from HMC import HMC


jax.config.update("jax_enable_x64", True) # avoid NaNs whilst calculating grads


# this is the density we want to sample from 
def density(x):
    return jnp.exp( -0.50*jnp.linalg.norm(x)**2 ) / jnp.sqrt(2*jnp.pi) 

# potential equals -log(density)
def potential(x):    
    return -jnp.log( density(x) )
                
########################
def main():
    # ensemble setup
    numDimensions = 2
    numParticles = 1
    temperature = 1/Boltzmann
    qStd = 1
    
    # integrator setup
    finalTime = 1
    stepSize = 0.1

    # PDF Setup
    mean = jnp.zeros(numDimensions)
    cov = np.random.uniform(size=(2, 2)) # random covariance matrix
    cov = np.dot(cov, cov.T) # variance must be positive
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)
    
    # HMC setup 
    numSamples = 100
    
    # generate and initialize ensemble
    ensemble1 = Ensemble(numDimensions, numParticles)
    
    # HMC algorithm
    hmcObject = HMC(ensemble1, finalTime, stepSize, densityFunc)
    hmcSamples, _ = hmcObject.getSamples(numSamples, temperature, qStd)




    
    
    # we test implementation on a 2D standard normal distribution
    ## theoretical resulst for 2D standard Gaussian distribution
    mean = np.zeros(numDimensions)
    normal = np.zeros_like(hmcSamples)

    for k in range(numSamples):
        normal[:, : , k] = np.random.multivariate_normal(mean, cov, size=numParticles).T
    

    # plot results normal distribution
    fig, ax = plt.subplots()
    ax.plot( hmcSamples[0, 0, :], hmcSamples[1, 0, :], label = "HMC", marker = '*', c='k', lw =0.2, ls ='-', markersize=4)
    ax.plot( normal[0, 0, :], normal[1, 0, :], label = "theoretical", marker = '.', c='r', lw =0.2, ls ='-', markersize=4)
    plt.title(r'2D standard Gaussian')
    plt.xlabel(r'$x_{1}$')
    plt.ylabel(r'$x_{2}$')
    plt.legend(loc='upper right')
    plt.show()
            
if __name__ == '__main__':
    hmcSamples = main()
    
    
