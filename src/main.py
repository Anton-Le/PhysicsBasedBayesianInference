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

#from scipy.constants import Boltzmann
Boltzmann = 1
#import argument parsing module
import argparse
from stepSizeSelection import dtProposal

import time
# import function used to initialize the
# distribution of positions.
from jax.scipy.stats import multivariate_normal

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

#temporary location for the ESS function

def effectiveSampleSize(weights: jnp.array, Z: float):
    return 1.0/ jnp.sum( ( jnp.exp(weights) /Z ) **2)

def resampleEnsemble(ensemble, Z: float, N_effective: int):
    print("Sorting and computing CDF")
    #compute the CDF of the ensemble weights
    cdf = jnp.cumsum( jnp.exp(ensemble.weights) / Z )
    #prepare the arrays for new particle positions and momenta
    q = jnp.zeros( ensemble.q.shape )
    p = jnp.zeros( ensemble.p.shape )
    weights = jnp.zeros( numParticles )
    # sort in descending order of magnitude of weights
    particleIndices = jnp.array(np.arange(numParticles))
    sortedParticleIndices = jax.lax.sort_key_val( jnp.exp(ensemble.weights) / Z, particleIndices, 0)[1]
    # copy into the new arrays - problematic since zip has broken contiguous arrays up
    q_collected = ensemble.q[sortedParticleIndices[:N_effective], :]
    p_collected = ensemble.p[sortedParticleIndices[:N_effective], :]
    w_collected = ensemble.weights[ sortedParticleIndices[:N_effective] ]
    q = jax.lax.dynamic_update_slice(q, q_collected, (0,0) )
    p = jax.lax.dynamic_update_slice(p, p_collected, (0,0) )
    weights = jax.lax.dynamic_update_slice(weights, w_collected, (0,))
    Z_new = jnp.sum( jnp.exp(weights) )
    cdf_updated = jnp.cumsum( jnp.exp(weights) / Z_new ) 
    #total energy
    print("Filling up the ensemble")
    #iterate over the remaining slots and draw particles from the ~original~ reduced ensemble
    for pId in range( ensemble.numParticles - N_effective ):
        u = np.random.uniform(0,1)
        srcParticleIdx = jnp.argmin( cdf_updated < u)
        q = q.at[N_effective + pId].set( q[srcParticleIdx] ) # ensemble.q[ srcParticleIdx ] )
        p = p.at[N_effective + pId].set( p[srcParticleIdx] ) # ensemble.p[ srcParticleIdx ] )

    #set the ensemble data
    ensemble.q = jnp.copy(q)
    ensemble.p = jnp.copy(p)
    #ensemble.initWeights = -jnp.ones( ensemble.numParticles ) * jnp.log( ensemble.numParticles )
    return ensemble;



if __name__ == "__main__":
    # select the platform
    platform = "gpu"

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
    data = json.load(open("CoinToss.data.json"))
    modelDataDictionary = {
        "c1": np.array(data["c1"]),
        "c2": np.array(data["c2"]),
    }
    p1_reference = float(data["p1"])
    p2_reference = float(data["p2"])

    model = coin_toss
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    statModel = statisticalModel(model, (), modelDataDictionary)

    # Define run-time parameters (to be acquired from command line later
    parser = argparse.ArgumentParser(description="Model testing parameter parser")
    parser.add_argument('numParticles', metavar='N', type=int, default=2, help="Number of particles (Markov Chains)")
    parser.add_argument("t_final", metavar="t", type=float, default=0.1, help="Total integration time")
    parser.add_argument("dt", metavar="dt", type=float, default=0.01, help="Step size")
    parser.add_argument("temp", metavar="T", type=float, default=1, help="Temperature in units of k_B")
    parser.add_argument("degenFraction", metavar="dFrac", type=float, default=0.5, help="Fraction of particles below which to resample.")
    parser.add_argument("thermSteps", metavar="tSteps", type=int, default=1, help="Number of thermalization steps.")
    inputArguments = parser.parse_args()
    numParticles = inputArguments.numParticles
    numDimensions = 2  # fetch from the model!
    temperature = inputArguments.temp / Boltzmann
    qStd = 1
    stepSize = inputArguments.dt
    finalTime = inputArguments.t_final

    tSteps = inputArguments.thermSteps
    particleFraction = inputArguments.degenFraction

    #Define the degeneracy threshold of particles
    N_threshold = int( np.rint(numParticles * particleFraction) )
    #random_seed = 1234
    random_seed = int( np.round(time.time() ) )
    rng_key = jax.random.PRNGKey(random_seed)
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

    dtSizes = dtProposal(ensemble, statModel.potential, stepSize )
    avgDt = jnp.average(dtSizes)
    
    print(f"Mean parameters after initialisation \n", initialEstimate)
    print("Averaged step size: ", avgDt)
    print(
        f"Mean parameters after initialisation, transformed: \n",
        statModel.converter.toArray(
            statModel.constraint_fn(statModel.converter.toDict(initialEstimate))
        ),
    )

    # HMC algorithm
    hmcObject = HMC(
        finalTime,
        float(avgDt),
        initialPositionDensityFunc,
        potential=statModel.potential,
        gradient=statModel.grad
    )
    # SMC loop
    #jax.profiler.start_trace("/mnt/TGT/JAXProfiles")
    for smcStep in range(tSteps):
        print("[SMC loop] step ", smcStep)
        #propagate the ensemble using HMC
        ensemble = hmcObject.propagate_ensemble(ensemble)
        # compute mean and partition function
        meanParameter, Z = ensemble.getWeightedMean()
        print("Partition function: ", Z[0])
        # SMC, determine effective sample size
        N_effective = effectiveSampleSize( ensemble.weights, Z[0] )
        print("Effective sample size: ", N_effective)
        # Print the HMC estimate
        print("Arithmetic mean parameter: ", meanParameter)
        resultVector = statModel.converter.toArray(
                statModel.constraint_fn(statModel.converter.toDict(meanParameter))
            )
        print(
                f"Mean parameters after HMC, transformed: \n",
                resultVector,
        )
        # Threshold, resample, perform next step SMC sampling
        if N_effective <= N_threshold:
            print("Resampling")
            resampleEnsemble(ensemble, Z[0], int(np.rint(N_effective)) )
        #SMC loop END
    #resultVector.block_until_ready()
    #jax.profiler.stop_trace()
    #final approximation
    meanParameter, Z = ensemble.getWeightedMean()
    print("Effective sample size: ", N_effective)
    # Print the HMC estimate
    print("Arithmetic mean parameter: ", meanParameter)
    resultVector = statModel.converter.toArray(
                statModel.constraint_fn(statModel.converter.toDict(meanParameter))
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
    print("Analytic bias of coin 1: ", p1_reference)
    print("Absolute error: ", abs(p1 - p1_reference))
    print("Relative error: ", abs(p1 - p1_reference) / p1_reference)

    print("Bias of coin 2: ", p2)
    print("Analytic bias of coin 2: ", p2_reference)
    print("Absolute error: ", abs(p2 - p2_reference))
    print("Relative error: ", abs(p2 - p2_reference) / p2_reference)
    print("Obtained samples: \n", ensemble.q)
    meanParameter, Z = ensemble.getWeightedMean()
    print(f"Mean parameters after HMC: \n", meanParameter)
