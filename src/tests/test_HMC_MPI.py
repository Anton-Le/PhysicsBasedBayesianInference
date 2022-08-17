#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 12:44:20 2022

@author: bruno, thomas
"""
import sys

# setting path
sys.path.append("../")

import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from jax.scipy.stats import multivariate_normal
import matplotlib as mpl
import jax
from HMC import HMC
from mpi4py import MPI, rc
from HMC_MPI import HMC_MPI

rc.threaded = True
rc.thread_level = "funneled"
comm = MPI.COMM_WORLD
rank = comm.Get_rank ()
size = comm.Get_size ()
print(rank)

def init():
    seed = 6
    numParticles = 12 # per ensemble
    numDimensions = 2

    # PDF Setup
    mean = jnp.ones(numDimensions) * 5
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)
    potentialFunc = lambda q: -multivariate_normal.logpdf(q, mean, cov=cov)

    numIterations = 30

    simulTime = 1
    stepSize = 0.01        
    temperature = 1 / Boltzmann 
    qStd = 3
    mass = jnp.ones(numParticles)

    hmcObject = HMC(
        numDimensions,
        simulTime,
        stepSize,        
        temperature, 
        qStd,
        densityFunc,
        potential=potentialFunc,
        method='Stormer-Verlet'
    )

    print(f'{size=}')
    print(f'{numParticles=}')
    print(f'{numDimensions=}')
    print(f'{numIterations=}')

    samples = HMC_MPI(hmcObject, numIterations, mass, seed)
    print(samples)

if __name__ == '__main__':
    init()

