#!/usr/bin/env python3

import sys

# setting path
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import k as boltzmannConst
from scipy.stats import norm
from ensemble import Ensemble
import jax
import jax.numpy as jnp


def boltzmannDistribution(velocity, temperature, mass):
    """Velocity boltzman distribution (for magnitude of velocity)
    https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#Distribution_function
    """
    mOver2kt = mass / (2 * boltzmannConst * temperature)
    prefactor1 = (mOver2kt / np.pi) ** (3 / 2)
    prefactor2 = 4 * np.pi * velocity ** 2
    return prefactor1 * prefactor2 * np.exp(-(velocity ** 2) * mOver2kt)


def test_init():
    numDimensions = 4
    numParticles = 100
    temperature = 10
    seed = 10

    ensemble = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )

    # expected output
    q_expected = np.zeros(numDimensions)
    p_expected = np.zeros(numDimensions)
    m_expected = 1.0  # 0.0
    w_expected = 0.0

    q1, p1, m1, w1 = ensemble.particle(10)

    assert np.all(q_expected == q1), "Unexpected position values"
    assert np.all(p_expected == p1), "Unexpected momentum values"
    assert np.all(m_expected == m1), "Unexpected mass values"
    assert np.all(w_expected == w1), "Unexpected p'bility weight values"


def main():
    numDimensions = 4
    numParticles = 100
    temperature = 10
    seed = 10

    ensemble1 = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )

    # expected output
    qExp = np.zeros(numDimensions)
    pExp = np.zeros(numDimensions)
    mExp = 1.0
    wExp = 0.0

    test_init()

    try:
        _ = ensemble1.particle(numParticles + 1)
        assert False, "No IndexError"
        print("Test 2 Failed")
    except IndexError as error:
        print(error)
        print("Test 2 Passed \n")

    print("Testing initial velocities follow boltzman distribution.")

    numDimensions2 = 3
    numParticles2 = 1000
    temperature2 = 300
    constMass = 1e-27
    mass2 = np.ones(numParticles2) * constMass
    boltzmannDist2 = lambda velocity: boltzmannDistribution(
        velocity, temperature2, constMass
    )

    ensemble2 = Ensemble(
        numDimensions2, numParticles2, temperature2, jax.random.PRNGKey(seed)
    )

    ensemble2.mass = mass2
    ensemble2.setMomentum()
    ensemble2.setPosition(3)

    momentum = ensemble2.p
    momentumMagnitudes = np.linalg.norm(momentum, axis=1)
    velocityMagnitudes = momentumMagnitudes / mass2
    vLinspace = np.linspace(0, max(velocityMagnitudes), 100)
    freq = boltzmannDist2(vLinspace)

    ensemble2.setMomentum()
    momentum2 = ensemble2.p

    assert (momentum2 != momentum).all(), "Momenta not randomly set"

    plt.hist(velocityMagnitudes, bins=30, density=True)
    plt.plot(vLinspace, freq)
    plt.show()


if __name__ == "__main__":
    main()
