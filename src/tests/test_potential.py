import sys

# setting path
sys.path.append("../")

from potential import harmonicPotentialND
from ensemble import Ensemble
import jax.numpy as jnp


def test_potential():
    springConsts = jnp.array([2, 3])
    harmonicPotential1 = lambda q: harmonicPotentialND(q, springConsts)

    ensemble = Ensemble(2, 10, 1, [0, 10])
    ensemble.mass = jnp.ones(10)

    print("We expect a potential of 2*3^2 + 3*4^2 / 2 =18+48 /2=33 at (3, 4)")
    ensemble.q = ensemble.q.at[0].set(jnp.array([3.0, 4.0]))
    expectedPot = 33
    calculatedPot = harmonicPotential1(ensemble.q[0])
    print(
        "Calculated potential at (3, 4) with springConsts (2, 3): ",
        calculatedPot,
    )
    assert calculatedPot == expectedPot, "ERROR! Incorrect potential value"


if __name__ == "__main__":
    test_potential()
