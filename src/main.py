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

if __name__=='__main__':
    # select the platform
    platform = "cpu"

    numpyro.set_platform(platform)
    # Print run-time configuration infromation
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
    model = coin_toss
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    statModel = statisticalModel(model, (), modelDataDictionary)

    # Define run-time parameters (to be acquired from command line later
    numParticles = 40
    numDimensions = 2 # fetch from the model!
    temperature = 1 / Boltzmann
    qStd = 1
    stepSize = 0.001
    finalTime = 0.1
    random_seed = 1234
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
    initialEstimate = np.zeros(numDimensions)
    Z = 0.0
    for particleId in range(numParticles):
        q, _, _, w = ensemble.particle(particleId)
        initialEstimate += w*q;
        Z += w;
    initialEstimate /= Z;
    print("Mean parameters after initialisation: \n", initialEstimate)

    print("Mean parameters after initialisation, transformed: \n", statModel.converter.toArray(statModel.constraint_fn(statModel.converter.toDict(initialEstimate))) )
    
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
    meanParameter = np.mean( ensemble.q, axis=0)
    print("Mean parameters after HMC: \n", meanParameter )
    print("Mean parameters after HMC, transformed: \n", statModel.converter.toArray(statModel.constraint_fn(statModel.converter.toDict(meanParameter))) )
