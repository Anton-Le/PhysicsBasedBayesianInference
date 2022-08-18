#!/usr/bin/env python3

import sys

# setting path
sys.path.append("../")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import k as boltzmannConst
from scipy.stats import norm
from ensemble import Ensemble


def boltzmannDistribution(velocity, temperature, mass):
    """Velocity boltzman distribution (for magnitude of velocity)
    https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#Distribution_function
    """
    mOver2kt = mass / (2 * boltzmannConst * temperature)
    prefactor1 = (mOver2kt / jnp.pi) ** (3 / 2)
    prefactor2 = 4 * jnp.pi * velocity**2
    return prefactor1 * prefactor2 * jnp.exp(-(velocity**2) * mOver2kt)


def test1():
    numDimensions = 4
    numParticles = 100
    seed = 10
    temperature = 1

    ensemble1 = Ensemble(
        numParticles, 
        numDimensions, 
        temperature, 
        jax.random.PRNGKey(seed)
        )

    # expected output
    qExp = jnp.zeros(numDimensions)
    pExp = jnp.zeros(numDimensions)
    mExp = 1

    q1, p1, m1 = ensemble1.particle(10)

    if (
        (qExp == q1).all()
        and (pExp == p1).all()
        and (mExp == m1).all()
    ):
        print("Test 1 passed.")

    try:
        _ = ensemble1.particle(numParticles + 1)
        print("Test 2 Failed")
    except IndexError as error:
        print(error)
        print("Test 2 Passed \n")



def test2():
    print("Testing initial velocities follow boltzman distribution.")

    numDimensions2 = 3
    numParticles2 = 1000
    seed = 10

    temperature2 = 300
    constMass = 1e-27
    mass2 = jnp.ones(numParticles2) * constMass
    boltzmannDist2 = lambda velocity: boltzmannDistribution(
        velocity, temperature2, constMass
    )

    ensemble2 = Ensemble(
        numParticles2, 
        numDimensions2, 
        temperature2, 
        jax.random.PRNGKey(seed)
        )
    ensemble2.mass = mass2
    ensemble2.setMomentum()
    ensemble2.setPosition(3)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter3D(*ensemble2.q) # check q
    # ax.scatter3D(*(ensemble2.p)) # check p
    # plt.show()
    momentum = ensemble2.p
    momentumMagnitudes = jnp.linalg.norm(momentum, axis=1)
    velocityMagnitudes = momentumMagnitudes / mass2
    vLinspace = jnp.linspace(0, max(velocityMagnitudes), 100)
    freq = boltzmannDist2(vLinspace)

    plt.hist(velocityMagnitudes, bins=30, density=True)
    plt.plot(vLinspace, freq)
    plt.show()


    print('check key is updated:')
    ensemble2.setMomentum()
    momentum_2 = ensemble2.p
    assert (momentum != momentum_2).all()


    ensemble2.setPosition(3)
    q_1 = ensemble2.q
    ensemble2.setPosition(3)
    q_2 = ensemble2.q
    assert (q_2 != q_1).all()
    print('passed')


if __name__ == "__main__":
    test1()
    test2()
