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
import jax.numpy as jnp
import numpy as np
import json

from matplotlib import pyplot as plt

# from scipy.constants import Boltzmann
Boltzmann = 1
from jax.scipy.stats import multivariate_normal
from ensemble import Ensemble
import matplotlib as mpl
from matplotlib.colors import LogNorm
import jax
from HMC_Python import HMC_reference
from HMC import HMC

from CoinToss import coin_toss
from converters import Converter
from potential import harmonicPotentialND, statisticalModel

jax.config.update("jax_enable_x64", True)
# seed the MT1337 generator
np.random.seed(1234)


def harmonicForce(q, q0, k):
    return k * (q - q0)


def test():
    # ensemble setup

    numParticles = 6
    numDimensions = 2
    temperature = 10 / Boltzmann
    seed = 10
    key = jax.random.PRNGKey(seed)

    ensemble = Ensemble(numDimensions, numParticles, temperature, key)

    # load model data and set up the statistical model
    # Load the observed outcomes and the reference biases
    data = json.load(open("../CoinToss.data.json"))
    modelDataDictionary = {
        "c1": np.array(data["c1"]),
        "c2": np.array(data["c2"]),
    }
    p1_reference = float(data["p1"])
    p2_reference = float(data["p2"])

    model = coin_toss
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    statModel = statisticalModel(model, (), modelDataDictionary)

    # PDF Setup
    mean = jnp.ones(numDimensions) * 2
    mean2 = jnp.ones(numDimensions) * -3
    cov = jnp.array([[4, -3], [-3, 4]])  # random covariance matrix
    densityFunc = lambda q: multivariate_normal.pdf(
        q, mean, cov=cov
    ) + multivariate_normal.pdf(q, mean2, cov=cov)
    springConsts = np.arange(1, 1 + numDimensions) ** 2
    print("k = ", springConsts)
    potentialFunc = statModel.potential

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
        potential=statModel.potential,
        gradient=statModel.grad,
    )

    hmcJAX = HMC(
        stepSize * numStepsPerIteration,
        stepSize,
        densityFunc,
        potential=statModel.potential,
        gradient=statModel.grad,
    )

    # set positions and momenta
    qStd = 3
    ensemble.setPosition(qStd)
    ensemble.setMomentum()
    # copy ensemble to the one rune by JAX
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
    hmcWeightsJAX = jnp.zeros((numIterations, numParticles))

    for i in range(numIterations):
        print("Step block ", i)
        ensemble = hmcObject.propagate_ensemble(ensemble)
        jaxEnsemble = hmcJAX.propagate_ensemble(jaxEnsemble)
        # copy particle data out
        hmcSamples = hmcSamples.at[i].set(ensemble.q)
        hmcSamplesJAX = hmcSamplesJAX.at[i].set(jaxEnsemble.q)
        hmcWeightsJAX = hmcWeightsJAX.at[i].set(jaxEnsemble.weights)
    # transform the position values from the unconstrained domain
    # to the model's domain
    #    for it in range(numIterations):
    #        for p in range(numParticles):
    # hmcSamplesJAX = hmcSamplesJAX.at[it,p].set(statModel.converter.toArray(
    #        statModel.constraint_fn(statModel.converter.toDict(hmcSamplesJAX[it,p]))
    #            ))
    # hmcSamples = hmcSamples.at[it,p].set(statModel.converter.toArray(
    #            statModel.constraint_fn(statModel.converter.toDict(hmcSamples[it,p]))
    #            ))

    # plot the resulting trajectories within a contour plot of the potential
    fig, ax = plt.subplots()
    pathCmap = plt.cm.get_cmap("Set1")
    for particleNum in range(numParticles):
        ax.plot(
            hmcSamples[:, particleNum, 0],
            hmcSamples[:, particleNum, 1],
            marker="*",
            color=pathCmap(particleNum),
            label="{:d}".format(particleNum + 1),
            lw=0.2,
            ls="-",
            markersize=4,
        )
        ax.plot(
            hmcSamplesJAX[:, particleNum, 0],
            hmcSamplesJAX[:, particleNum, 1],
            color=pathCmap(particleNum),
            marker="o",
            # label="JAX: {:d}".format(particleNum + 1),
            lw=0.2,
            ls=":",
            markersize=2,
        )
    # contour plot

    x = np.linspace(-7, 7, 100)
    y = np.linspace(-5, 5, 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    q = np.dstack((x_mesh, y_mesh))
    z = np.zeros_like(x_mesh)
    # iterate
    for row in range(x_mesh.shape[1]):
        for col in range(x_mesh.shape[0]):
            z[row, col] = potentialFunc(jnp.array(q[row, col]))
    # for visualisation purposes we require the mapped grid
    xMapped = np.zeros_like(x)
    yMapped = np.zeros_like(y)
    for i in range(len(xMapped)):
        q = np.array([x[i], 0])
        qMapped = statModel.converter.toArray(
            statModel.constraint_fn(statModel.converter.toDict(q))
        )
        xMapped[i] = qMapped[0]
    for i in range(len(yMapped)):
        q = np.array([0, y[i]])
        qMapped = statModel.converter.toArray(
            statModel.constraint_fn(statModel.converter.toDict(q))
        )
        yMapped[i] = qMapped[1]
    # x_mesh, y_mesh = np.meshgrid(xMapped, yMapped)
    cmap = plt.get_cmap("jet")
    norm = LogNorm(vmin=z.min(), vmax=z.max())
    contour = ax.contourf(x_mesh, y_mesh, z, cmap=cmap, levels=40)
    fig.colorbar(contour)
    # plot the true parameters
    ax.scatter([0.0], [1.1], s=50, marker="x", color="m")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$x_{1}$", fontsize="x-large")
    ax.set_ylabel(r"$x_{2}$", fontsize="x-large")
    ax.legend(title="Particle", loc="lower left")
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    fig.savefig("HMC_CT_unconstrainedPositions_wrongJacobiMatrix.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    test()
