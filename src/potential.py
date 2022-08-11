#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 14 2022
@author: thomas

File contain potentials & function to calculate n-body gravitational force.

"""

import numpy as np
import jax.numpy as jnp
from scipy.constants import G as gravConst
from numbers import Real  # check if variable is number
from scipy.optimize import approx_fprime
import numpyro
import jax

def harmonicPotentialND(q, springConsts):
    """
    @description:
        Returns harmonic potential at points given by q.
    @parameters:
        q (ndarray): numDimensions x numParticles array of positions
        springConsts (ndarray): numDimensions array of spring constants
    """

    return 0.5 * jnp.dot(springConsts, q**2)


def getAccelNBody(q, mass, i):
    """
    @description:
        Calculate acceleration of i th particle in N-body system.

    @parameters:
        q (ndarray): numDimensions x numParticles array of positions
        i (int): index
        mass (ndarray): numParticles array of masses
    """
    iQ = q[:, [i]]
    iMass = mass[i]

    # remove i th particle
    qReduced = np.delete(q, i, axis=1)
    massReduced = np.delete(mass, i)

    r = qReduced - iQ

    denom = np.linalg.norm(r, axis=0) ** 3

    accelArray = gravConst * massReduced * r / denom

    return np.sum(accelArray, axis=1)


def gravitationalPotential(r1, r2, mass1, mass2):
    """
    @description:
        Returns potential between two masses.

    @parameters:
        r1 (ndarray): Position vector of first mass.
        r2 (ndarray): Position vector of second mass.
        mass1 (ndarray): First mass
        mass2 (ndarray): Second mass
    """
    r = r1 - r2
    distance = np.sqrt(np.dot(r, r))
    return gravConst * mass1 * mass2 / distance


def nBodyPotential(q, mass, shape=None):
    """
    @description:
        Calculate n body potential for gravitational force.
    @parameters:
        q (ndarray): numDimensions x numParticles array of positions
        mass (ndarray): numParticles array of masses
        shape (array-like): original shape of q in case q is 1D array.
    """

    if shape != None:  # approx_fprime requires q to be 1D array - Fixed here:
        q = q.reshape(shape)

    remainingParticles = q.shape[1]
    potential = 0
    countedParticles = 0
    for particleNum_i in range(q.shape[1]):
        remainingParticles -= 1
        countedParticles += 1

        for particleNum_j in range(remainingParticles):

            potential += gravitationalPotential(
                q[:, particleNum_i],
                q[:, countedParticles + particleNum_j],
                mass[particleNum_i],
                mass[countedParticles + particleNum_j],
            )

    return potential


def nBodyForce(q, mass):
    """
    @description:
        Calculate n body potential for gravitational force.

    @parameters:
        q (ndarray): numDimensions x numParticles array of positions
        mass (ndarray): numParticles array of masses
    """

    outputShape = q.shape
    gradient = approx_fprime(
        np.ravel(q), nBodyPotential, 1.49e-08, mass, outputShape
    )
    gradient = gradient.reshape(outputShape)
    return -gradient


def getForceArray(q, potentialFunc, dq):
    """
    @description:
        Calculate forces at each position vector.

    @parameters:
        q (ndarray): numDimensions x numParticles array of positions
        potentialFunc (func):
        dq (float): Step to evaluate derivatives
    """

    force = np.zeros_like(q)

    for i in range(q.shape[1]):  # for each particle
        force[:, i] = -approx_fprime(q[:, i], potentialFunc, 1e-8)

    return force


def noPotential(q):
    return 0

def statisticalModelPotential(model, position, converter, modelArgs, modelKwargs):
    """
    @description:
        The function is used to provide the potential value
        for a stochastic model using unconstrained positions.
    @parameters:
        model ( function ) : probabilistic model
        position (ndarray): numDimensions x numParticles array of positions
        converter ( Converter ): Converter object to convert arrays to dictionaries
    """
    return -numpyro.infer.util.log_density(
                model,
                modelArgs, 
                modelKwargs,
                converter.toDict(position),
                )[0]

def statisticalModelGradient(model, position, converter, modelArgs, modelKwargs):
    dictGrad = jax.grad(
                lambda x: numpyro.infer.util.log_density(
                model,
                modelArgs,
                modelKwargs,
                x)[0]
                )( converter.toDict(position) )
    return converter.toArray(dictGrad)

