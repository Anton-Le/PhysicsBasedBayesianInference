#!/usr/bin/env python3

from ensemble import Ensemble
import numpy as np
from scipy.constants import G as gravConst
from numbers import Real # check if variable is number
from scipy.optimize import approx_fprime

def harmonicPotentialND(q, springConsts):
	if q.shape[0] != len(springConsts):
		raise ValueError('k must be 1D array corresponding with spring constant \
			in 3 dimensions.')
	
	q_T = q.T

	return np.sum((0.5 * springConsts * q_T ** 2), axis=1)

def gravitationalPotential(r1, r2, mass1, mass2):
	r = r1 - r2
	distance = np.sqrt(np.dot(r, r))
	return gravConst * mass1 * mass2 / distance


def nBodyPotential(q, mass, shape=None):
	''' Calculate n body potential for gravitational force.

	Gravitation interaction is a function of a 2 1D arrays (position of 2 
	particles)
	
	q1 and q2 are given as separate arguments to allow the gradient of the
	function with respect to each particles position to be more easily found.

	'''
	if shape != None: # approx_fprime requires q to be 1D array - Fixed here:
		q = q.reshape(shape)

	remainingParticles = q.shape[1]
	potential = 0
	countedParticles = 0
	for particleNum_i in range(q.shape[1]):
		remainingParticles -= 1
		countedParticles += 1

		for particleNum_j in range(remainingParticles):
			print(particleNum_i, countedParticles + particleNum_j)

			potential += gravitationalPotential(
				q[:, particleNum_i], 
				q[:, countedParticles + particleNum_j],
				mass[particleNum_i],
				mass[countedParticles + particleNum_j]
				)
			print(potential)

	return potential

def nBodyForce(q, mass):
	''' dq should be a 1D array with length equal to the number of dimensions, 
	or a scalar.
	
	'''
	outputShape = q.shape
	gradient = approx_fprime(np.ravel(q), nBodyPotential, 1.49e-08, 
		mass, outputShape)
	gradient = gradient.reshape(outputShape)
	return -gradient


def noPotential(q):
	return 0


# numDimensions, numParticles = 2, 100
# harmonicPotential12 = lambda q: harmonicPotentialND(q, np.array([1, 2]))
# ensemble = Ensemble(numDimensions, numParticles, harmonicPotential12)
# ensemble.initializeThermal(np.ones(numParticles), 300, 33)

# print(ensemble.potential(ensemble.q).shape)


test_ensemble = Ensemble(numDimensions=3, numParticles=2, potential=None)
test_ensemble.mass = np.array([2, 3])

test_ensemble.q = np.array([[3, 5],
							[10, 20], 
							[10, 21]])
# distance between (2, 10, 15) and (4, 20, 30) is sqrt(15)
# grav potential should be:
expPotential = gravConst * 2 * 3 / 15
calcPotential = nBodyPotential(test_ensemble.q,
	test_ensemble.mass)

# grav force should be 2.67
expForce = gravConst * 2 * 3 / 225
forceVectors = nBodyForce(test_ensemble.q, mass=test_ensemble.mass)
print('Potential:')
print(f'Expected: {expPotential:.10g}, Observed: {calcPotential:.10g}')

print('Force:')
print(f'Vector: {forceVector}')
print('Magnitude')
print(f'Expected: {expForce:.5g} ,Observed: {np.linalg.norm(forceVector[:, 0]):.5g}')