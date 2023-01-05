#!/usr/bin/env python3

import sys

# setting path
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import k as boltzmannConst
from scipy.stats import norm
import numpyro, os

from ensemble import Ensemble
from potential import harmonicPotentialND
from stepSizeSelection import dtProposal

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"


def test_stepSizeProposal_basic():
    """
    This function implements a basic test of the step-size
    proposal functionality by computing the average
    step size for 100 particles using the harmonic potential.
    We only check whether the value is physical (>0).
    """
    numDimensions = 4
    numParticles = 100
    temperature = 1
    seed = 10
    springConsts = jnp.ones(numDimensions)
    potential = lambda q: harmonicPotentialND(q, springConsts)

    ensemble = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )
    ensemble.setPosition()
    ensemble.setMomentum()

    dt = jnp.average( dtProposal(ensemble, potential) )
    assert dt > 0, "Step size is unphysical"


def analyticStepSize(q, p, mass,T, K, p0):
    a = jnp.dot( K**2, q**2)
    b = jnp.dot(p, K*p)
    c = 2 * mass * boltzmannConst * T *np.log(1/p0)
    dt1 = np.sqrt( c/(a+b/mass ) )
    dt2 = -dt1
    return (dt1, dt2)

def test_stepSizeProposal_HO():
    """
    This function uses an analytic way of determining 
    the largest step size dt and compares it to the
    numerically obtained values.
    """

    numDimensions = 2
    numParticles = 2
    temperature = 10/boltzmannConst
    seed = 10
    springConsts = jnp.ones(numDimensions)#jnp.array( np.arange(1,numDimensions+1) )
    potential = lambda q: harmonicPotentialND(q, springConsts)
    # initialise an ensemble, we determine dt given these data
    ensemble = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )
    # set positions
    ensemble.q = ensemble.q.at[0].set( [0.0, 0] )
    ensemble.q = ensemble.q.at[1].set([1.0, 0])
    # set momenta
    ensemble.p = ensemble.p.at[0].set([1.0, 0]) 
    ensemble.p = ensemble.p.at[1].set( [-1.0,0]) 

    stepSizes = []
    E0 = []
    E1 = []
    for pId in range(numParticles):
        particle = ensemble.particle(pId)
        q, p, mass, _ = particle
#        print("Particle ", pId)
#        print(q, p, mass)
        E0.append(jnp.dot(p,p)/(2*mass) + potential(q)  )
#        print("grad(V)(q) = ", grad(potential)(q) )
        E1.append(jnp.dot(p - springConsts*q,p - springConsts*q)/(2*mass) + potential(q + p/mass) )
        dt1, dt2 = analyticStepSize(q, p, mass, temperature, springConsts, 0.5)
        stepSizes.append(  float(dt1) if (float(dt1) > float(dt2) ) else float(dt2) )
    print("Analytic step sizes: ", stepSizes)
    numStepSizes = dtProposal(ensemble, potential)
    print("Numeric step sizes: ", numStepSizes)
    # Improvement required!
    for i in range(numParticles):
            assert (numStepSizes[i] <= stepSizes[i]), "Implausible step size for particle {:d}".format(i)

def test_stepSizeProposal_T_convergence():
    """
    This function checks whether analytic and numeric results
    converge to the same value with increasing temperature (at a fixed N).

    The expected result is that the difference in step sizes will decrease
    with increasing temperature.
    """
    print("Testing step size convergence with increasing temperature")
    numDimensions = 4
    numParticles = 1000
    temperature = 1.0/boltzmannConst
    seed = 10
    springConsts = jnp.array( np.arange(1,numDimensions+1) )
    potential = lambda q: harmonicPotentialND(q, springConsts)
    # initialise an ensemble, we determine dt given these data
    ensemble = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )
    ensemble.setPosition()
    ensemble.setMomentum()

    avgNumStepSizes = []
    avgTheoStepSizes = []
    # loop over a set of temperatures
    temperatureSchedule = np.logspace(0, 4, num=5) / boltzmannConst
    for T in temperatureSchedule:
            #update ensemble
            ensemble = Ensemble(
                            numDimensions, numParticles, T, jax.random.PRNGKey(seed)
                            )
            ensemble.setPosition()
            ensemble.setMomentum()
            #compute the analytic step sizes
            avgStepSize = 0.0
            for pId in range(numParticles):
                particle = ensemble.particle(pId)
                q, p, mass, _ = particle
                dt1, dt2 = analyticStepSize(q, p, mass, T, springConsts, 0.5)
                avgStepSize += (  float(dt1) if (float(dt1) > float(dt2) ) else float(dt2) ) / numParticles
            avgTheoStepSizes.append(avgStepSize)
            # compute the numeric step sizes
            numStepSizes = dtProposal(ensemble, potential)
            avgNumStepSizes.append( float( jnp.average(numStepSizes) ) )
    # ealuate differences
    stepSizeDifference = np.array( [a - b for (a,b) in zip(avgTheoStepSizes, avgNumStepSizes )]  )
    relErr = abs(stepSizeDifference) / avgTheoStepSizes
    print("Theoretical averages: ", avgTheoStepSizes)
    print("Numeric averages: ", avgNumStepSizes)
    print(relErr)
    assert np.all(relErr[:-1] > relErr[1:]) , "Relative error does not decrease with temperature increase"
    print("Temperature convergence PASSED!")

def test_stepSizeProposal_N_convergence():
    """
    This function checks whether analytic and numeric results
    converge to the same value with increasing number of particles (at fixed T).

    The expected result is that the difference in step sizes will decrease.
    """
    print("Testing step size convergence with increasing ensemble size")
    numDimensions = 4
    temperature = 10000/boltzmannConst
    seed = 10
    springConsts = jnp.array( np.arange(1,numDimensions+1) )
    potential = lambda q: harmonicPotentialND(q, springConsts)
    # initialise an ensemble, we determine dt given these data

    avgNumStepSizes = []
    avgTheoStepSizes = []
    # loop over a set of temperatures
    ensembleSize = np.logspace(1, 4, num=4, dtype=int)
    for N in ensembleSize:
            print("Testing with an ensemble with {:d} particles".format(N))
            numParticles = N
            #update ensemble
            ensemble = Ensemble(
                            numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
                            )
            ensemble.setPosition()
            ensemble.setMomentum()
            #compute the analytic step sizes
            avgStepSize = 0.0
            for pId in range(numParticles):
                particle = ensemble.particle(pId)
                q, p, mass, _ = particle
                dt1, dt2 = analyticStepSize(q, p, mass, temperature, springConsts, 0.5)
                avgStepSize += (  float(dt1) if (float(dt1) > float(dt2) ) else float(dt2) ) / numParticles
            avgTheoStepSizes.append(avgStepSize)
            # compute the numeric step sizes
            numStepSizes = dtProposal(ensemble, potential)
            avgNumStepSizes.append( float( jnp.average(numStepSizes) ) )

    # ealuate differences
    stepSizeDifference = np.array( [a - b for (a,b) in zip(avgTheoStepSizes, avgNumStepSizes )]  )
    relErr = abs(stepSizeDifference) / avgTheoStepSizes
    print(avgTheoStepSizes)
    print(avgNumStepSizes)
    print("Relative errors: ", relErr)
    assert np.all(relErr[:-1] > relErr[1:]) , "Relative error does not decrease for larger ensemble"
    print("Ensemble size convergence PASSED!")

def main():
    print("Testing step size proposal functionality")
    test_stepSizeProposal_basic()
    test_stepSizeProposal_HO()
    test_stepSizeProposal_T_convergence()
    test_stepSizeProposal_N_convergence()

if __name__ == "__main__":
    # select the platform
    platform = "cpu"

    numpyro.set_platform(platform)

    main()
