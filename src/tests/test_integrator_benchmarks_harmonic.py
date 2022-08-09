#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: thomas
"""
import sys

# setting path
sys.path.append("../")

from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
from potential import harmonicPotentialND
import numpy as np
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt
from jax import jit, pmap, grad
import itertools
import cProfile
import pstats


NUM_DIMENSIONS = 3
NUM_PARTICLES = 100
Q_STD = 10
MASS_VAL = 1
MASS = np.ones(NUM_PARTICLES) * MASS_VAL
V_PARTICLES = 10
TEMPERATURE = MASS_VAL * V_PARTICLES**2 / (2 * Boltzmann)  # T =mv^2 / 2 k_b

METHOD = "Stormer-Verlet"
STEP_SIZE = 0.1
FINAL_TIME = 10

NUM_ITERATIONS = 2

# create ensemble

springConsts = np.array((2.0, 3.0, 4.0))
harmonicPotential = lambda q: harmonicPotentialND(q, springConsts)
harmonicGradient = grad(harmonicPotential)
print(harmonicGradient(np.array([2.0, 2.0, 3.0])))

ensemble1 = Ensemble(NUM_DIMENSIONS, NUM_PARTICLES)
ensemble1.mass = MASS
ensemble1.setMomentum(TEMPERATURE)
ensemble1.setPosition(Q_STD)


# create integrator


if METHOD == "Leapfrog":
    integrator = Leapfrog(ensemble1, STEP_SIZE, FINAL_TIME, harmonicGradient)
elif METHOD == "Stormer-Verlet":
    integrator = StormerVerlet(
        ensemble1, STEP_SIZE, FINAL_TIME, harmonicGradient
    )
else:
    raise ValueError("Invalid method.")


# perform integration

points = np.zeros((NUM_DIMENSIONS, NUM_PARTICLES, NUM_ITERATIONS))

points[:, :, 0] = ensemble1.q[0]

profiler = cProfile.Profile()
profiler.enable()
q, p = integrator.integrate()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats("ncalls")
fileName = input("File name: ")
stats.dump_stats(f"{fileName}.log")

points[:, :, 1] = q


# plot points

fig = plt.figure()
ax = plt.axes(projection="3d")


for particleNum in range(NUM_PARTICLES):
    x = points[0, particleNum, :]
    y = points[1, particleNum, :]
    z = points[2, particleNum, :]
    print(x, y, z)
    ax.plot3D(x, y, z)

plt.show()
