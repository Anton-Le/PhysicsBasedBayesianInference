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

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"


def test_stepSizeProposal():
    numDimensions = 4
    numParticles = 100
    temperature = 1
    seed = 10
    springConsts = jnp.ones(numDimensions)
    print("Spring constants: ", springConsts )
    potential = lambda q: harmonicPotentialND(q, springConsts)

    ensemble = Ensemble(
        numDimensions, numParticles, temperature, jax.random.PRNGKey(seed)
    )
    ensemble.setPosition()
    ensemble.setMomentum()
    print("Shape of the positions: ", ensemble.q.shape)

    dt = dtProposal(ensemble, potential)
    print("Step size: ", dt)
    assert dt > 0, "Step size is unphysical"


def main():
    print("Testing step size proposal functionality")
    test_stepSizeProposal()


if __name__ == "__main__":
    # select the platform
    platform = "gpu"

    numpyro.set_platform(platform)

    main()
