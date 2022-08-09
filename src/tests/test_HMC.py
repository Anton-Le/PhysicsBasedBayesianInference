#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 12:44:20 2022

@author: thomas
"""
import sys

# setting path
sys.path.append("../")

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from jax.scipy.stats import multivariate_normal
import jax
from HMC import HMC


jax.config.update("jax_enable_x64", True) 


def test1():
    seed = 1000
    key = jax.random.PRNGKey(seed)
    numParticles = 4
    numDimensions = 2

    # PDF Setup
    mean = jnp.ones(numDimensions) * 5
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)
    potentialFunc = lambda q: -multivariate_normal.logpdf(q, mean, cov=cov)

    numIterations = 100
    subkeys = jax.random.split(key, numParticles)
    simulTime = 1
    stepSize = 0.01        
    temperature = 1 / Boltzmann 
    qStd = 3
    mass = jnp.ones(numParticles)

    # get samples from numpy 
    numpySamples = np.random.multivariate_normal(
        mean, cov, size=numParticles * numIterations
    )


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

    ax.plot(
        hmcSamples[0, 0],
        hmcSamples[0, 1],
        label="HMC",
        marker="*",
        c="k",
        lw=0.2,
        ls="-",
        markersize=4,
    )

    for particleNum in range(1, numParticles):
        ax.plot(
            hmcSamples[particleNum, 0],
            hmcSamples[particleNum, 1],
            marker="*",
            c="k",
            lw=0.2,
            ls="-",
            markersize=4,
        )

    ax.plot(
        numpySamples[:, 0],
        numpySamples[:, 1],
        label="Numpy",
        marker=".",
        c="r",
        lw=0.2,
        ls="",
        markersize=4,
    )

    plt.title(r"Guassian with mean (5, 5)")
    plt.xlabel(r"$x_{1}$")
    plt.ylabel(r"$x_{2}$")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    hmcSamples = test1()  # change to test1 if desired
