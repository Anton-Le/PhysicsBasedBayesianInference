import sys

# setting path
sys.path.append('../')

import os
import jax.numpy as jnp
from jax import grad, make_jaxpr
from integrator_pmap import Leapfrog, StormerVerlet
from potential import harmonicPotentialND
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

springConsts = jnp.array((4., 4.)) # must be floats to work with grad
harmonicPotential = lambda q: harmonicPotentialND(q, springConsts)
harmonicGradient = grad(harmonicPotential)

FINAL_TIME = jnp.pi * 1.5
STEP_SIZES = jnp.logspace(-8, -1, 10)




initialQ = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
initialP = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
mass = jnp.ones(3)

def harmonicOscillatorAnalytic(finalTime, springConsts, initialQ, initialP, mass):
    omega = jnp.sqrt(jnp.outer(1/mass, springConsts))
    print(omega)
    mass = mass[:, None]
    initialV = initialP / mass
    q = initialQ * jnp.cos(omega * finalTime) + initialV / omega * jnp.sin(omega * finalTime)
    v = - omega * initialQ * jnp.sin(omega * finalTime) + initialV * jnp.cos(omega * finalTime)
    return (q, v * mass)


q_ana, p_ana = harmonicOscillatorAnalytic(FINAL_TIME, springConsts, initialQ, initialP, mass)


errors = jnp.zeros(len(STEP_SIZES))
for method in [Leapfrog, StormerVerlet]:

    for i, stepSize in enumerate(STEP_SIZES):
        integrator = method(stepSize, FINAL_TIME, harmonicGradient)
        q_num, p_num = integrator.integrate(initialQ, initialP, mass)
        errors = errors.at[i].set( jnp.average(jnp.abs(q_num - q_ana)) )

    print(errors)
    plt.loglog(STEP_SIZES, errors, label=f'{integrator.__class__.__name__}')

print(errors)

print(initialQ)
print(q_ana)
print(q_num)

plt.legend()
plt.show()
