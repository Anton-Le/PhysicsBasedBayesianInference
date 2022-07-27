#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: bruno, thomas
"""


import sys

# setting path
sys.path.append("../")

from ensemble import Ensemble
from integrator import integrator_methods
import numpy as np


def harmonicPotential(q):
    return q


def freeParticleAnalytic(ensemble, numSteps, dt):
    time = numSteps * dt
    q = ensemble.q * time * ensemble.p / ensemble.mass
    return (q, ensemble.p)


def harmonicOscillatorAnalytic(ensemble, numSteps, dt, springConsts):
    omega = sqrt(springConsts / mass)
    initialV = ensemble.p / ensemble.mass
    # constants
    a = 0.5 * (ensemble.q - (1j / omega) * initialV)
    b = ensemble.q - a
    q = np.real(a * np.exp(1j * omega * time) + b * np.exp(-1j * omega * time))
    v = np.real(
        1j * omega * (a * np.exp(1j * omega * time) - b * np.exp(-1j * omega * time))
    )
    return (q, v * ensemble.mass)


def main():
    # integrator setup
    finalTime = 1
    stepSize = 0.01
    method = "SV"

    # ensemble setup
    numDimensions = 2
    numParticles = 2
    mass = 2 * np.random.uniform(0, 1, numParticles)

    temperature = 1
    q_std = 0.1

    # ensemble initialization
    ensemble1 = Ensemble(numDimensions, numParticles, harmonicPotential)
    ensemble1.initializeThermal(mass, temperature, q_std)

    # object of class integartor

    sol_num = integrator_methods(ensemble1, stepSize, finalTime)

    # actual solution for position and momenta
    sol_num.numerical_solution(method)

    q, p, f, a = sol_num.get_quantities()
    return (q, p, f, a)


if __name__ == "__main__":
    q, p, f, a = main()
