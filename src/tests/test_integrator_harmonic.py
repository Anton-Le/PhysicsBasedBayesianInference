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
from potential import harmonicPotentialND
from jax import grad, pmap
import numpy as np
import matplotlib.pyplot as plt
import itertools


springConsts = np.array((2.0, 3.0))  # must be floats to work with grad
harmonicPotential = lambda q: harmonicPotentialND(q, springConsts)
harmonicGradient = grad(harmonicPotential)


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
    numDimensions = 2  # must match len(springConsts)
    mass = 1
    temperature = 1000
    q_std = 10

    # integrator setup

    omega1stDimension = np.sqrt(springConsts[0] / mass)
    period1stDimension = (
        2 * np.pi / omega1stDimension
    )  # choose period to check validity of analytical solution.
    finalTime = period1stDimension  # After 1 (1st dimension) period positions/momenta should be the same in 1st dimension
    print(f"Duration: {finalTime}")

    mass = np.ones(numParticles) * mass

    # ensemble initialization
    ensemble1 = Ensemble(numParticles, numDimensions)
    ensemble1.mass = mass
    ensemble1.setPosition(q_std)
    ensemble1.setMomentum(temperature)
    q, p = ensemble1.q, ensemble1.p

    print("Initial conditions:")
    print(ensemble1.q[dimension])
    print(30 * "#")

    # object of class Integrator - CHANGE IF DESIRED
    if method == "Leapfrog":
        intMethod = Leapfrog
    elif method == "Stormer-Verlet":
        intMethod = StormerVerlet
    else:
        raise ValueError("Method must be 'Leapfrog' or 'Stormer-Verlet'")

    integrator = intMethod(stepSize, finalTime, harmonicGradient)

    q_num = np.zeros((numParticles, numDimensions))
    p_num = np.zeros_like(q_num)

    # actual solution for position and momenta
    for i in range(numParticles):
        q_num[i], p_num[i] = integrator.integrate(q[i], p[i], mass[i])

    numSteps = int(finalTime / stepSize)
    q_ana, p_ana = harmonicOscillatorAnalytic(
        ensemble1, finalTime, springConsts
    )

    print("Numeric Solution:")
    print(q_num[:, dimension])
    print(30 * "#")

    print("Analytic Solution:")
    print(q_ana[:, dimension])
    print(30 * "#")

    print(f'{q_num.shape=}')
    return np.abs(q_num[:, dimension] - q_ana[:, dimension])


def plotError():
    methods = ["Leapfrog", "Stormer-Verlet"]
    numParticles = 3  
    numDimensions = 1
    stepSizes = np.logspace(-4, -1, 5)
    logStepSizes = np.log10(stepSizes)
    errors = np.zeros((len(stepSizes), numParticles))

    fig, ax = plt.subplots()

    markers = itertools.cycle(("x"))

    for method in methods:

        marker = next(markers)
        color = next(ax._get_lines.prop_cycler)["color"]
        for j, stepSize in enumerate(stepSizes):
            errors[j, :] = harmonic_test(stepSize, numParticles, method)

        logErr = np.log10(errors)
        meanErr = np.mean(logErr, axis=1)
        stdErr = np.std(logErr, axis=1)

        ax.errorbar(
            logStepSizes,
            meanErr,
            yerr=stdErr,
            label=method,
            color=color,
            linestyle="",
            marker=marker,
            markersize=3,
            capsize=3,
            elinewidth=1,
        )

        for xz, yz in zip(logStepSizes, logErr):
            ax.scatter(
                [xz] * len(yz), yz, color=color, marker=marker, s=1.8, alpha=0.7
            )

    ax.set_title("Integrator Error vs Step-Size")
    ax.set_ylabel("Log Absolute Error")
    ax.set_xlabel("Log Step-Size")
    ax.legend()

    # fig.savefig('qErrorVsStepSize.png')
    plt.show()


if __name__ == "__main__":
    plotError()
    # harmonic_test(0.001, 1)


def freeParticleAnalytic(ensemble, numSteps, dt):
    time = numSteps * dt
    q = ensemble.q * time * ensemble.p / ensemble.mass

    return q, ensemble.p
