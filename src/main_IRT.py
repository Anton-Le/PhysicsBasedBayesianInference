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
import time
sys.path.append("./irt_2pl")
# Import the model and the converter
#from CoinToss import coin_toss
from irt_2pl import model as irt_model
from irt_2pl import parameters_info
from converters import Converter
from potential import statisticalModel
from ensemble import Ensemble
from HMC import HMC

from numpyro.handlers import seed

#from scipy.constants import Boltzmann
Boltzmann = 1
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

def outputEstimates(estimates: np.array, filenamePrefix: str, Nparticles: int, tFinal: float,\
                dt: float, temperature: float, resamplingFraction: float, SMCsteps: int, runtime=1.0):
        """
        Function used to output the list of estimated parameters into a
        text file with a name encoding all utilised parameters.
        """
        filenameBodyTemplate="N_{:d}_t_{:.2f}_dt_{:.2e}_T_{:.2f}_f_{:.2f}_Tsmc_{:d}"
        filename = filenamePrefix + "_" + filenameBodyTemplate.format(Nparticles, tFinal, dt, temperature,\
                        resamplingFraction, SMCsteps)+".txt"
        np.savetxt(filename, estimates, footer="runtime {} [s]".format(runtime) )


if __name__ == "__main__":
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
        data = json.load(open("irt_2pl/irt_2pl.data.json"))
        modelDataDictionary = {
            "I": int(data["I"]),
            "J": int(data["J"]),
            "y": np.array( data["y"], dtype=int).reshape( (data['yshape'][0], data['yshape'][1]) )
        }
    else:
        modelDataDictionary = None

    modelDataDictionary = comm.bcast(modelDataDictionary, root=0)
    model = irt_model
    # CAVEAT: model arguments are an empty tuple here, subject to change!
    statModel = statisticalModel(model, (), modelDataDictionary)

    # Define run-time parameters (to be acquired from command line later
    parser = argparse.ArgumentParser(description="Model testing parameter parser")
    parser.add_argument('numParticles', metavar='N', type=int, default=2, help="Number of particles (Markov Chains)")
    parser.add_argument("t_final", metavar="t", type=float, default=0.1, help="Total integration time")
    parser.add_argument("dt", metavar="dt", type=float, default=0.01, help="Step size")
    parser.add_argument("temp", metavar="T", type=float, default=1, help="Temperature in units of k_B")
    parser.add_argument("filePrefix", metavar="prefix", type=str, default="CT", help="Prefix to the file name.")
    inputArguments = parser.parse_args()
    #numParticles = 2**15
    numParticles = inputArguments.numParticles // size
    numDimensions = statModel.converter.vectorSize  # fetch from the model!
    #temperature = 0.1 / Boltzmann
    temperature = inputArguments.temp / Boltzmann
    qStd = 1
    #stepSize = 0.001
    stepSize = inputArguments.dt
    #finalTime = 0.1
    finalTime = inputArguments.t_final
    prefix = inputArguments.filePrefix
    tStart = time.time()
    random_seed = 1234
    #random_seed = int( np.round(time.time() ) )
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
        parameterEstimateHistory = np.zeros( (2, ensemble.numDimensions) )
        parameterEstimateHistory[0] = statModel.converter.toArray(
                statModel.constraint_fn(statModel.converter.toDict(initialMean))
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
        print("Transformed: ",statModel.constraint_fn(statModel.converter.toDict(finalMean)  ) )
        resultDict = statModel.constraint_fn(statModel.converter.toDict(finalMean)  )
        #manual copy of the data
        manuallyTransformedData = np.zeros_like(resultVector)
        keysequence = ['sigma_theta', 'theta', 'sigma_a', 'a', 'mu_b','sigma_b', 'b']
        offsetIdx = 0
        for key in keysequence:
            data = resultDict[key]
            print(offsetIdx)
            manuallyTransformedData[offsetIdx:offsetIdx + data.size] = np.copy(data)
            offsetIdx += data.size

        parameterEstimateHistory[1] = manuallyTransformedData # np.copy( np.array(resultVector) )
        tStop=time.time()
        outputEstimates(parameterEstimateHistory, prefix+"_HMC_"+platform, numParticles, finalTime, stepSize, temperature, 1.0, 1, tStop - tStart)
        # We assume that the data stored in the input file that is not part of the model data
        # are reference parameters
        a_ref = jnp.array(data['a'])
        b_ref = jnp.array(data['b'])
        theta_ref = jnp.array(data['theta'])
        AbsErr_theta = jnp.abs(theta_ref - resultDict['theta'])
        AbsErr_a = jnp.abs(a_ref - resultDict['a'])
        AbsErr_b = jnp.abs(b_ref - resultDict['b'])
        print("Max err a: ", jnp.max(AbsErr_a))
        print("Max err b: ", jnp.max(AbsErr_b))
        print("Max err theta: ", jnp.max(AbsErr_theta))
        RelErr_theta = jnp.abs( AbsErr_theta / theta_ref)
        RelErr_a = jnp.abs(AbsErr_a / a_ref)
        RelErr_b = jnp.abs(AbsErr_b / b_ref)
        print("Max rel. err a: ", jnp.max( RelErr_a ))
        print("Max rel. err b: ", jnp.max( RelErr_b ))
        print("Max rel. err theta: ", jnp.max( RelErr_theta ))
