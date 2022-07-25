
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: thomas
"""
import sys

# setting path
sys.path.append('../')


from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import itertools


# create ensemble

NUM_DIMENSIONS = 3
numParticles = 3
temperature = 1000
qStd = 1
earthMass = 5.972e24
sunMass = 1.989e30
moonMass = 7.34e22
mass = np.array([earthMass, sunMass, moonMass])

ensemble1 = Ensemble(NUM_DIMENSIONS, numParticles, None)
ensemble1.mass = mass

# we expect the gravitational force to be (-3, -4)/125 * G at (3, 4)
ensemble1.q[:, 0] = np.array([1.52e11, 0, 0])
ensemble1.p[:, 0] = np.array([0, 29800 , 0]) * earthMass
ensemble1.q[:, 2] = np.array([1.52e11, 3.844e8, 0])
ensemble1.p[:, 2] = np.array([0, 29800, 1022]) * moonMass

# create integrator

method = 'Stormer-Verlet'
stepSize = 600
finalTime = 3600

if method == 'Leapfrog':
	integrator = Leapfrog(ensemble1, stepSize, finalTime)
elif method == 'Stormer-Verlet':
	integrator = StormerVerlet(ensemble1, stepSize, finalTime)
else:
	raise ValueError('Not valid integrator method.')


# run integrator multiple times

totalTime = 365 * 24 * 3600 # 3 years

numIterations = int(totalTime / finalTime)

points = np.zeros((NUM_DIMENSIONS, numParticles, numIterations))

for i in range(numIterations):
	q, p = integrator.integrate()
	points[:, :, i] = q


# plot points 

fig = plt.figure()
ax = plt.axes(projection='3d')

for particleNum in range(numParticles):
	x = points[0, particleNum, :]
	y = points[1, particleNum, :]
	z = points[2, particleNum, :]
	print(x, y, z)
	ax.plot3D(x, y, z)

fig.savefig(f'{method}SolarSystem.png')
plt.show()

