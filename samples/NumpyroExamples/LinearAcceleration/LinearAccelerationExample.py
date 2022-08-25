import numpyro
import jax
import numpy as np
import json

numpyro.set_platform("cpu")


# Print run-time configuration infromation
print(f"jax version: {jax.__version__}")
print(f"numpyro version: {numpyro.__version__}")
print(f"jax target backend: {jax.config.FLAGS.jax_backend_target}")
print(f"jax target device: {jax.lib.xla_bridge.get_backend().platform}")
cpus = jax.devices("cpu")
print("Available CPUs:")
print(cpus)

# Import the model (function)
from LinearAcceleration import linear_accel

from collections import OrderedDict

# Load the observed outcomes and the reference biases
data = json.load(open("LinearMotion.data.json"))

# print the raw data for visual inspection
print("Loaded data (raw): ")
print(data)
print("=" * 20)

# store the user-provided 'true' values of the biases

# Prepare the kernel and run
model = linear_accel
nuts_kernel = numpyro.infer.NUTS(model)
# use 500 MCMC chains as 'burn-in' and 500 to obtain the actual parameter smples
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=1024)
rng_key = jax.random.PRNGKey(0)
mcmc.run(
    rng_key,
    t=np.array(data["t"]),
    z=np.array(data["z"]),
    sigmaObs=float(data["sigmaObs"]),
    extra_fields=("potential_energy",),
)

# Retrieve the samples of the parameters (p1, p2) from the MCMC object
# each parameter will be an array of `num_samples` values.
params = mcmc.get_samples()
print("Parameters obtained using NumPyros MCMC kernels (raw data): ")
print(params)

# Retrieve the parameters from the output

h = np.array(params["h"])
v0 = np.array(params["v0"])
g = np.array(params["g"])

# Since this is Markov-Chain monte Carlo with MH proposal
# We may use simple averaging to obtain the parameters
print("Mean height: ", np.average(h))
print("Mean initial velocity: ", np.average(v0))
print("Mean acceleration: ", np.average(g))
