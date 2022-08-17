#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 12:44:20 2022

@author: bruno, thomas
"""
import sys

# setting path
sys.path.append("../")

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from jax.scipy.stats import multivariate_normal
import matplotlib as mpl
import jax
from HMC import HMC
from mpi4py import MPI, rc


jax.config.update("jax_enable_x64", True) 


def test1():
    seed = 6
    key = jax.random.PRNGKey(seed)
    numParticles = 2
    numDimensions = 2

    # PDF Setup
    mean = jnp.ones(numDimensions) * 5
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)
    potentialFunc = lambda q: -multivariate_normal.logpdf(q, mean, cov=cov)

    numIterations = 30

    subkeys = jax.random.split(key, numParticles)
    simulTime = 1
    stepSize = 0.01        
    temperature = 1 / Boltzmann 
    qStd = 3
    mass = jnp.ones(numParticles)


    # get samples from HMC

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

    hmcSamples, _ = hmcObject.getSamples(numIterations, mass, subkeys)

    fig, ax = plt.subplots()


    colors=['black', 'red', 'blue', 'green']
    for particleNum in range(numParticles):
        ax.plot(
            hmcSamples[particleNum, 0],
            hmcSamples[particleNum, 1],
            marker="*",
            color=colors[particleNum],
            label=particleNum+1,
            lw=0.2,
            ls="-",
            markersize=4,
        )


    # contour plot
    x = np.linspace(mean[0]-7, mean[0]+7)
    y = np.linspace(mean[1]-7, mean[1]+7)
    x_mesh, y_mesh = np.meshgrid(x, y)
    q = np.dstack((x_mesh, y_mesh))
    z = densityFunc(q)


    cmap = plt.get_cmap("Pastel1").copy()

    contour = ax.contour(
        x_mesh,
        y_mesh,
        z,
        cmap=cmap,
        zorder=0,
    )



    plt.title(r"Guassian with mean (5, 5)")
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    plt.legend(title='HMC Branch', loc="upper right")
    plt.show()


if __name__ == "__main__":
    hmcSamples = test1()  # change to test1 if desired
