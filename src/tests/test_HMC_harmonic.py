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
from matplotlib.colors import LogNorm
import jax
from HMC_Python import HMC_reference
from potential import harmonicPotentialND
from HMC import HMC

jax.config.update("jax_enable_x64", True)
#seed the MT1337 generator
np.random.seed(1234)

def harmonicForce(q, q0, k):
    return k*(q-q0);

def test():
    # ensemble setup

    numParticles = 4
    numDimensions = 2
    temperature = 10 / Boltzmann
    seed = 10
    key = jax.random.PRNGKey(seed)

    ensemble = Ensemble(numDimensions, numParticles, temperature, key)

    # PDF Setup
    mean = jnp.ones(numDimensions) * 2
    mean2 = jnp.ones(numDimensions) * -3
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov) + multivariate_normal.pdf(q, mean2, cov=cov)
    springConsts = np.arange(1, 1+numDimensions)**2
    print("k = ", springConsts)
    potentialFunc = lambda q: harmonicPotentialND(q - mean, springConsts) * harmonicPotentialND(q - mean2, springConsts)
    #potentialFunc = lambda q: -multivariate_normal.logpdf(q, mean, cov=cov) -multivariate_normal.logpdf(q, mean2, cov=cov)
    analyticGradient = lambda q: harmonicForce(q, mean, springConsts) + harmonicForce(q, mean2, springConsts)
    # HMC setup
    simulTime = 1.0
    numIterations = 30
    numStepsPerIteration = 10
    stepSize = simulTime / (numIterations * numStepsPerIteration)

    # to propagate in blocks change final time
    hmcObject = HMC_reference(
        stepSize * numStepsPerIteration,
        stepSize,
        densityFunc,
        potential=potentialFunc
    )

    hmcJAX = HMC(
        stepSize * numStepsPerIteration,
        stepSize,
        densityFunc,
        potential=potentialFunc
    )


    # set positions and momenta
    qStd = 3
    ensemble.setPosition(qStd)
    ensemble.setMomentum()
    #copy ensemble to the one rune by JAX
    jaxEnsemble = ensemble

    hmcSamples = jnp.zeros(
        (
            numIterations,
            numParticles,
            numDimensions,
        )
    )
    hmcSamplesJAX = jnp.zeros(
        (
            numIterations,
            numParticles,
            numDimensions,
        )
    )

    for i in range(numIterations):
        print("Step block ", i)
        ensemble = hmcObject.propagate_ensemble(ensemble)
        jaxEnsemble=hmcJAX.propagate_ensemble(jaxEnsemble)
        #copy particle data out
        hmcSamples = hmcSamples.at[i].set(ensemble.q)
        hmcSamplesJAX = hmcSamplesJAX.at[i].set(jaxEnsemble.q)


    fig, ax = plt.subplots()

    for particleNum in range(numParticles):
        ax.plot(
            hmcSamples[:, particleNum, 0],
            hmcSamples[:, particleNum, 1],
            marker="*",
            label="PY: {:d}".format(particleNum + 1),
            lw=0.2,
            ls="-",
            markersize=4,
        )
        ax.plot(
            hmcSamplesJAX[:, particleNum, 0],
            hmcSamplesJAX[:, particleNum, 1],
            marker="o",
            label="JAX: {:d}".format(particleNum + 1),
            lw=0.2,
            ls=":",
            markersize=2,
        )
    # contour plot

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    q = np.dstack((x_mesh, y_mesh))
    z = np.zeros_like(x_mesh)
    #iterate
    for row in range(x_mesh.shape[1]):
        for col in range(x_mesh.shape[0]):
            z[row, col] = potentialFunc( jnp.array(q[row, col] ) )

    cmap = plt.get_cmap("jet")
    norm = LogNorm(vmin=z.min(), vmax=z.max() )
    print(z.min(), z.max())
    contour = ax.contour(
        x_mesh,
        y_mesh,
        z,
        cmap=cmap,
        levels=20,
        norm=norm
    )

    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.legend(title="HMC Branch", loc="upper right")
    plt.show()


if __name__ == "__main__":
    test()
