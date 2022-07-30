import numpyro
import jax
import numpy as np
import json

numpyro.set_platform("cpu")


# Lets print some information about our system
print(f"jax version: {jax.__version__}")
print(f"numpyro version: {numpyro.__version__}")
print(f"jax target backend: {jax.config.FLAGS.jax_backend_target}")
print(f"jax target device: {jax.lib.xla_bridge.get_backend().platform}")
cpus = jax.devices("cpu")
print("Available CPUs:")
print(cpus)

# Load the model
from eight_schools import eight_schools

# Load the model data
data = json.load(open("eight_schools.data.json"))

# Prepare the kernel and run
model = eight_schools
nuts_kernel = numpyro.infer.NUTS(model)
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=10)
rng_key = jax.random.PRNGKey(0)
mcmc.run(
    rng_key,
    data["J"],
    np.array(data["sigma"]),
    y=np.array(data["y"]),
    extra_fields=("potential_energy",),
)

# You can access the samples by:
params = mcmc.get_samples()
print(params)
# print(mcmc.get_extra_fields())

# Lets try more specific functions
random_seed = 1234
rng_key = jax.random.PRNGKey(random_seed)
log_d = numpyro.infer.util.log_density(model, [], data, params)[0]
print("Log density: ", log_d)
