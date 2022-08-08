import sys

# setting path
sys.path.append('../')


import jax.numpy as jnp
from jax import grad
from integrator import Leapfrog
from potential import harmonicPotentialND

springConsts = jnp.array((2., 3.)) # must be floats to work with grad
harmonicPotential = lambda q: harmonicPotentialND(q, springConsts)
harmonicGradient = grad(harmonicPotential)

STEP_SIZE, FINAL_TIME = 0.001, 1

integrator = Leapfrog(STEP_SIZE, FINAL_TIME, harmonicGradient)

q = jnp.array([1., 2.],) #[3., 4.], [5, 6]])
p = jnp.array([1., 2.],)# [3., 4.], [5, 6]])
mass = jnp.ones(1)

q, p = integrator.integrate(q, p, mass)

print(q, p)