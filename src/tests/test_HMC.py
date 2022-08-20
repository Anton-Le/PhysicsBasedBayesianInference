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
from ensemble import Ensemble
import matplotlib as mpl
import jax
from HMC import HMC


jax.config.update("jax_enable_x64", True) 


def test():
    # ensemble setup

    numParticles = 3
    numDimensions = 2
    temperature = 1 / Boltzmann
    seed = 10
    key = jax.random.PRNGKey(seed)  

    ensemble = Ensemble(numParticles, numDimensions, temperature, key)
    

    # PDF Setup
    mean = jnp.ones(numDimensions) * 5
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)
    potentialFunc = lambda q: -multivariate_normal.logpdf(q, mean, cov=cov)  

    
    # HMC setup
    simulTime = 1
    stepSize = 0.01   


    hmcObject = HMC(
        simulTime,
        stepSize,        
        densityFunc,
        potential=potentialFunc,
        method='Stormer-Verlet'
    )

    # set positions

    qStd = 3
    ensemble.setPosition(3)

    numIterations = 100

    hmcSamples = jnp.zeros((numIterations, numParticles, numDimensions,))

    for i in range(numIterations):
        ensemble.setMomentum() # numParticles!

        ensemble = hmcObject.propagate_ensemble(ensemble)

        hmcSamples = hmcSamples.at[i].set(ensemble.q)


    fig, ax = plt.subplots()


    for particleNum in range(numParticles):
        ax.plot(
            hmcSamples[:, particleNum, 0],
            hmcSamples[:, particleNum, 1],
            marker="*",
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

    cmap = plt.get_cmap("Pastel1")

    contour = ax.contour(
        x_mesh,
        y_mesh,
        z,
        cmap=cmap,
        zorder=0,
    )


    ax.set_title(r"Guassian with mean (5, 5)")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.legend(title='HMC Branch', loc="upper right")

    plt.show()



if __name__ == "__main__":
    test()

