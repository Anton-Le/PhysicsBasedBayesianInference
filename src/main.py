#!/bin/env python3
"""
This file implements the main routine of fitting a model
to data using HMC.

Date: 17.08.2022
Author: Anton Lebedev
"""

import sys
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
# import function used to initialize the
# distribution of positions.
from jax.scipy.stats import multivariate_normal
from mpi4py import MPI, rc
import mpi4jax
rc.threaded = True
rc.thread_level = "funneled"
comm = MPI.COMM_WORLD
rank = comm.Get_rank ()
size = comm.Get_size ()


if __name__=='__main__':
    # select the platform
    platform = "cpu"

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
        data = json.load(
            open("CoinToss.data.json")
        )
        modelDataDictionary = {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])}
    else:
        modelDataDictionary = None
    model = coin_toss
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    modelDataDictionary = comm.bcast(modelDataDictionary, root=0)
    statModel = statisticalModel(model, (), modelDataDictionary)

    # Define run-time parameters (to be acquired from command line later
    numParticles = 40
    numDimensions = 2 # fetch from the model!
    temperature = 1 / Boltzmann
    qStd = 1
    stepSize = 0.001
    finalTime = 0.1
    random_seed = 1234
    random_seed = random_seed + rank
    rng_key = jax.random.PRNGKey(random_seed)
    seed(model, rng_key)
    # Set up the initial distribution of particles
    mean = jnp.zeros(numDimensions)
    cov = np.random.uniform(size=(numDimensions, numDimensions))  # random covariance matrix
    cov = np.dot(cov, cov.T)  # variance must be positive
    initialPositionDensityFunc = lambda q: multivariate_normal.pdf(q, mean, cov=cov)

    ensemble = Ensemble(numParticles, numDimensions, temperature, rng_key)
    # set the weights and momenta
    ensemble.setPosition(qStd)
    ensemble.setMomentum()
    ensemble.setWeights(statModel.potential)

    # compute initial mean values
    initialEstimate, initialZ = ensemble.getWeightedMean()

    initialZSum = mpi4jax.reduce(initial_z, mpi4py.MPI.SUM, root=0)

    initialWeightedSums = mpi4jax.gather(initialEstimate * initialZ, comm=comm, root=0)

    initialWeightedMean = initialWeightedSum / initialZSum

    print(f"Mean parameters after initialisation ({rank=}) \n", initialEstimate)

    print(f"Mean parameters after initialisation, transformed ({rank=}): \n", statModel.converter.toArray(statModel.constraint_fn(statModel.converter.toDict(initialEstimate))) )
    
    # HMC algorithm
    hmcObject = HMC(
        finalTime, 
        stepSize, 
        initialPositionDensityFunc, 
        potential=statModel.potential, 
        gradient=statModel.grad
    )

    ensemble = hmcObject.propagate_ensemble(ensemble)
    print("Obtained samples: \n", ensemble.q)
    meanParameter = ensemble.getWeightedMean()
    print(f"Mean parameters after HMC ({rank=}): \n", meanParameter )
    print(f"Mean parameters after HMC, transformed ({rank=}): \n", statModel.converter.toArray(statModel.constraint_fn(statModel.converter.toDict(meanParameter))) )
