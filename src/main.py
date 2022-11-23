#!/bin/env python3
"""
This file implements the main routine of fitting a model
to data using HMC.

Date: 17.08.2022
Author: Anton Lebedev
"""

import sys, os
import numpyro, jax
import jax.numpy as jnp
import numpy as np
import json

# Import the model and the converter
from CoinToss import coin_toss
from converters import Converter
from potential import statisticalModel
from ensemble import Ensemble
from HMC import HMC

from numpyro.handlers import seed

from scipy.constants import Boltzmann
#import argument parsing module
import argparse

# import function used to initialize the
# distribution of positions.
from jax.scipy.stats import multivariate_normal

import mpi4jax
from mpi4py import MPI, rc
rc.threaded = True
rc.thread_level = "funneled"
comm = MPI.COMM_WORLD
rank = comm.Get_rank ()
size = comm.Get_size ()

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

def getWeightedMeanStd(estimate, Z):
    '''
    Get weighted mean from multiple ensembles and error.
    '''
    estimateTimesZ, _ = mpi4jax.gather(estimate * Z, root=0)
    estimate, _ = mpi4jax.gather(estimate, root=0)
    Z, _ = mpi4jax.gather(Z, root=0)
    
    if rank == 0:
        #totalZ = jnp.sum(Z)
        #weightedMean = jnp.sum(estimateTimesZ, axis=0) / totalZ
        weightedMean = jnp.average(estimate, axis=0, weights=Z)
        std = jnp.sqrt(
            jnp.average((estimate-weightedMean)**2, axis=0, weights=Z)
            )

        return weightedMean, std
    return (None, None)


if __name__ == "__main__":
    # select the platform
    platform = "gpu"

    numpyro.set_platform(platform)
    # Print run-time configuration infromation

    if rank == 0:

        print(f"jax version: {jax.__version__}")
        print(f"numpyro version: {numpyro.__version__}")
        print(f"jax target backend: {jax.config.FLAGS.jax_backend_target}")
        print(f"jax target device: {jax.lib.xla_bridge.get_backend().platform}")
        devices = jax.devices(platform)
        print("Available devices:")
        print(devices)

        # load model data and set up the statistical model
        # Load the observed outcomes and the reference biases
        data = json.load(open("CoinToss.data.json"))
        modelDataDictionary = {
            "c1": np.array(data["c1"]),
            "c2": np.array(data["c2"]),
        }
        p1_reference = float(data["p1"])
        p2_reference = float(data["p2"])
    else:
        modelDataDictionary = None

    modelDataDictionary = comm.bcast(modelDataDictionary, root=0)
    model = coin_toss
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    statModel = statisticalModel(model, (), modelDataDictionary)

    # Define run-time parameters (to be acquired from command line later
    parser = argparse.ArgumentParser(description="Model testing parameter parser")
    parser.add_argument('numParticles', metavar='N', type=int, default=2, help="Number of particles (Markov Chains)")
    parser.add_argument("t_final", metavar="t", type=float, default=0.1, help="Total integration time")
    parser.add_argument("dt", metavar="dt", type=float, default=0.01, help="Step size")
    parser.add_argument("temp", metavar="T", type=float, default=1, help="Temperature in units of k_B")
    inputArguments = parser.parse_args()
    #numParticles = 2**15
    numParticles = inputArguments.numParticles
    numDimensions = 2  # fetch from the model!
    #temperature = 0.1 / Boltzmann
    temperature = inputArguments.temp / Boltzmann
    qStd = 1
    #stepSize = 0.001
    stepSize = inputArguments.dt
    #finalTime = 0.1
    finalTime = inputArguments.t_final
    random_seed = 1234
    rng_key = jax.random.PRNGKey(random_seed+rank)
    seed(model, rng_key)

    # Set up the initial distribution of particles
    mean = jnp.zeros(numDimensions)
    cov = np.random.uniform(
        size=(numDimensions, numDimensions)
    )  # random covariance matrix
    cov = np.dot(cov, cov.T)  # variance must be positive
    initialPositionDensityFunc = lambda q: multivariate_normal.pdf(
        q, mean, cov=cov
    )

    ensemble = Ensemble(numDimensions, numParticles, temperature, rng_key)
    # set the weights and momenta
    ensemble.setPosition(qStd)
    ensemble.setMomentum()
    ensemble.setWeights(statModel.potential)

    # compute initial mean values
    initialEstimate, initialZ = ensemble.getWeightedMean()

    initialMean, initialStd = getWeightedMeanStd(initialEstimate, initialZ)

    if rank == 0:
        print(f"Mean parameters after initialisation \n", f'{initialMean} +- {initialStd}')

        print(
            f"Mean parameters after initialisation, transformed: \n",
            statModel.converter.toArray(
                statModel.constraint_fn(statModel.converter.toDict(initialMean))
            ),
        )

    # HMC algorithm
    hmcObject = HMC(
        finalTime,
        stepSize,
        initialPositionDensityFunc,
        potential=statModel.potential,
        gradient=statModel.grad,
    )

    ensemble = hmcObject.propagate_ensemble(ensemble)

    meanParameter = ensemble.getArithmeticMean()
    print("Arithmetic mean parameter: ", meanParameter)
    finalMean, finalStd = getWeightedMeanStd(meanParameter, 1.0)
    if rank == 0:
        print(f"Mean parameters after HMC: \n", f'{finalMean} +- {finalStd}')
        resultVector = statModel.converter.toArray(
                statModel.constraint_fn(statModel.converter.toDict(finalMean))
            )
        print(
            f"Mean parameters after HMC, transformed: \n",
            resultVector,
        )
        p1 = resultVector[0]
        p2 = resultVector[1]
        # Since this is Markov-Chain monte Carlo with MH proposal
        # We may use simple averaging to obtain the parameters
        print("Bias of coin 1: ", p1)
        print("Absolute error: ", abs(p1 - p1_reference))
        print("Relative error: ", abs(p1 - p1_reference) / p1_reference)

        print("Bias of coin 2: ", p2)
        print("Absolute error: ", abs(p2 - p2_reference))
        print("Relative error: ", abs(p2 - p2_reference) / p2_reference)
