#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: bruno
"""
import sys

# setting path
sys.path.append("../")

from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
from scipy.constants import Boltzmann
from potential import harmonicPotentialND
import jax, numpyro
import jax.profiler
from jax import grad, pmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
jax.config.update("jax_enable_x64", True)
numpyro.set_platform("gpu")
#jax.profiler.start_server(9999)

springConsts = np.arange(1,11,1) #np.array((2.0, 3.0))  # must be floats to work with grad
harmonicPotential = lambda q: harmonicPotentialND(q, springConsts)
harmonicGradient = lambda q: 2*springConsts*q#grad(harmonicPotential)


def harmonicOscillatorAnalytic(ensemble, finalTime, springConsts):
    omega = np.outer(1 / ensemble.mass, springConsts)
    omega = np.sqrt(omega)
    initialV = ensemble.p / ensemble.mass[:, None]
    q = ensemble.q * np.cos(omega * finalTime) + initialV / omega * np.sin(
        omega * finalTime
    )
    v = -omega * ensemble.q * np.sin(omega * finalTime) + initialV * np.cos(
        omega * finalTime
    )

    return (q, v * ensemble.mass[:, None])

def harmonic_test(stepSize, numParticles, method):
    dimension = 0  # choose dimension to print positions
    # ensemble variables
    numDimensions = 10  # must match len(springConsts)
    mass = 1
    temperature = 1
    q_std = 10

    # integrator setup

    omega1stDimension = np.sqrt(springConsts[-1] / mass)
    period1stDimension = (
        2 * np.pi / omega1stDimension
    )  # choose period to check validity of analytical solution.
    finalTime = 2*period1stDimension  # After 1 (1st dimension) period positions/momenta should be the same in 1st dimension


    mass = jnp.ones(numParticles) * mass

    # ensemble initialization
    ensemble1 = Ensemble(numParticles, numDimensions)
    ensemble1.mass = mass
    ensemble1.setPosition(q_std)
    ensemble1.setMomentum(temperature)
    q, p = ensemble1.q, ensemble1.p


    # object of class Integrator - CHANGE IF DESIRED
    if method == "Leapfrog":
        intMethod = Leapfrog
    elif method == "Stormer-Verlet":
        intMethod = StormerVerlet
    else:
        raise ValueError("Method must be 'Leapfrog' or 'Stormer-Verlet'")

    integrator = intMethod(stepSize, finalTime, harmonicGradient)

    q_num = jnp.zeros((numParticles, numDimensions))
    p_num = jnp.zeros_like(q_num)
    jax.device_put(q_num)
    jax.device_put(p_num)
    jax.device_put(q)
    jax.device_put(p)
    jax.device_put(mass)
    # actual solution for position and momenta
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        q_num, p_num = integrator.pintegrate(q, p, mass)
        q_num.block_until_ready()
        p_num.block_until_ready()

    numSteps = int(finalTime / stepSize)
    q_ana, p_ana = harmonicOscillatorAnalytic(
        ensemble1, finalTime, springConsts
    )

    return np.abs(q_num[:, dimension] - q_ana[:, dimension])


def runProfileCase():
    methods = ["Leapfrog", "Stormer-Verlet"]
    numParticles = 10
    stepSizes = np.logspace(-7, -1, 14)
    logStepSizes = np.log10(stepSizes)
    errors = np.zeros((len(stepSizes), numParticles))


    # run one step size
    errors[0, :] = harmonic_test(stepSizes[-3], numParticles, methods[0] )


if __name__ == "__main__":
    runProfileCase()


def freeParticleAnalytic(ensemble, numSteps, dt):
    time = numSteps * dt
    q = ensemble.q * time * ensemble.p / ensemble.mass

    return q, ensemble.p


