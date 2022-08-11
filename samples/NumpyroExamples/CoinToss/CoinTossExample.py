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
from CoinToss import coin_toss

from collections import OrderedDict

# Load the observed outcomes and the reference biases
data = json.load(open("CoinToss.data.json"))

# print the raw data for visual inspection
print("Loaded data (raw): ")
print(data)
print("=" * 20)

# store the user-provided 'true' values of the biases
p1_reference = float(data["p1"])
p2_reference = float(data["p2"])

# Prepare the kernel and run
model = coin_toss
nuts_kernel = numpyro.infer.NUTS(model)
# use 500 MCMC chains as 'burn-in' and 500 to obtain the actual parameter smples
mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=500, num_samples=500)
rng_key = jax.random.PRNGKey(0)
mcmc.run(
    rng_key,
    c1=np.array(data["c1"]),
    c2=np.array(data["c2"]),
    extra_fields=("potential_energy",),
)

# Retrieve the samples of the parameters (p1, p2) from the MCMC object
# each parameter will be an array of `num_samples` values.
params = mcmc.get_samples()
print("Parameters obtained using NumPyros MCMC kernels (raw data): ")
print(params)

# Retrieve the parameters from the output

p1 = np.array(params["p1"])
p2 = np.array(params["p2"])

# Since this is Markov-Chain monte Carlo with MH proposal
# We may use simple averaging to obtain the parameters
print("Bias of coin 1: ", np.mean(p1))
print("Absolute error: ", abs(np.mean(p1) - p1_reference))
print("Relative error: ", abs(np.mean(p1) - p1_reference) / p1_reference)

print("Bias of coin 2: ", np.mean(p2))
print("Absolute error: ", abs(np.mean(p2) - p2_reference))
print("Relative error: ", abs(np.mean(p2) - p2_reference) / p2_reference)

# Lets try more specific functions
random_seed = 1234
rng_key = jax.random.PRNGKey(random_seed)
# log_d = numpyro.infer.util.log_density(model, [], data, params)[0]
# Compute the log(p(x|c)) - natural logarithm of the conditional probability density
# **NOTE** `log_density` returns an extensive dict, of which the first element
# is the numerical approximation of the log density.

log_p = numpyro.infer.util.log_density(
    model,
    (),
    {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])},
    {"p1": np.mean(p1), "p2": np.mean(p2)},
)
print("Log density given the determined parameters: ", log_p[0])
# Do the same with reference values
log_p = numpyro.infer.util.log_density(
    model,
    (),
    {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])},
    {"p1": p1_reference, "p2": p2_reference},
)
print("Log density given the reference parameters: ", log_p[0])

# Finally an example on how to compute gradients

# jax.grad requires a 1-parameter function, hence we pass a lambda fucntion with the data and model 'baked in'
# and allow only the passing of a dictionary for the parameter values.
# `grad` also expects numerical output (jax arrays), so we limit the output to element 0 only.
dictGrad = jax.grad(
    lambda x: numpyro.infer.util.log_density(
        model, (), {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])}, x
    )[0]
)({"p1": np.mean(p1), "p2": np.mean(p2)})
print("Gradient (as a dictionary) for the computed parameters: ", dictGrad)
dictGrad = jax.grad(
    lambda x: numpyro.infer.util.log_density(
        model, (), {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])}, x
    )[0]
)({"p1": p1_reference, "p2": p2_reference})
print("Gradient (as a dictionary) for the reference parameters: ", dictGrad)

# As expected for the bias values that were chosen to generate the data the gradient vanishes.

# Testing transformations from the real line onto the support of the distribution

testParamDict = {'p1':-10.0,'p2':0.5}

# Map/constrain the parameters onto the support of the model distribution

constrainedParams = numpyro.infer.util.constrain_fn(
                model,
                (),
                {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])},
                testParamDict
                )

print(" Unconstrained parameters: ", testParamDict)
print(" Constrained parameters: ", constrainedParams)
# Testing the computation of the jacobian

Jacobian = jax.jacfwd(
                lambda x: numpyro.infer.util.constrain_fn(
        model,
        (),
        {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])},
        x
       )
                )(testParamDict)

print("Jacobian of the parameter transform: ", Jacobian)

# Compute the gradient on the support of the model
# and then transform it to the real line

constrainedGradient = jax.grad(
    lambda x: numpyro.infer.util.log_density(
        model, (),
        {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])},
        x
    )[0]
)( constrainedParams )

print("Gradient on constrained domain: ", constrainedGradient)
# Transform in accordance with the notes: grad_u = J^t * grad_c

unconstrainedGradient = {}.fromkeys( constrainedGradient.keys() )

# iterate over the keys of the unconstrained

for key in unconstrainedGradient.keys():
    val = 0.0
    for c_key in constrainedGradient.keys():
        val += constrainedGradient[c_key] * Jacobian[c_key][key]
    unconstrainedGradient[key] = val;

print("Gradient on unconstrained domain: ", unconstrainedGradient)

# Collect the parameters of the model that will be optimized over

with numpyro.handlers.seed(rng_seed=1):
    trace = numpyro.handlers.trace(coin_toss).get_trace(**{"c1": np.array(data["c1"]), "c2": np.array(data["c2"])})

optParamsAndShapes = OrderedDict()

for paramName in trace.keys():
        paramProperties = trace[paramName]
        if (paramProperties['type'] is 'sample') and (paramProperties['is_observed'] is False):
                optParamsAndShapes[paramName] = paramProperties['value'].size

print("Parameters to optimize over :", optParamsAndShapes.keys() )
print("Their shapes: ", optParamsAndShapes.values() )

# define a function that will take a vector/array and the above dict as inputs
# and will produce a dict that can be passed to the gradient/density

def arrayToDict(paramShapes, vec):
    print(type(paramShapes))
    size = 0;
    for item in paramShapes.values():
        size += item;
    #ensure shapes check out
    assert(size == vec.size)
    paramData = OrderedDict().fromkeys( paramShapes.keys() );
    # stupid way
    arrayIdx = 0
    for paramName in paramShapes.keys():
        paramData[paramName] = vec[arrayIdx:arrayIdx + paramShapes[paramName] ].copy()
        arrayIdx = paramShapes[paramName]
    return paramData

automappedParamDict = arrayToDict(optParamsAndShapes, np.array([p1_reference, p2_reference]) )
print("Automatically mapped parameters: ", automappedParamDict)

dictGrad = jax.grad(
    lambda x: numpyro.infer.util.log_density(
        model, (), {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])}, x
    )[0]
)(automappedParamDict)
print("Gradient obtained with it: ", dictGrad)

# define an inverse function that will collect the values from a dict
# into an array

def dictToArray(paramShapes, paramDict):
    size = 0
    for item in paramShapes.values():
        size += item;
    vec = np.zeros(size);
    arrayIdx = 0
    for paramName in paramShapes.keys():
        vec[arrayIdx:arrayIdx + paramShapes[paramName] ] = paramDict[paramName].copy()
        arrayIdx = paramShapes[paramName]
    return vec

vectorialGradient = dictToArray(optParamsAndShapes, dictGrad)
print("Gradient mapped to a vector: ", vectorialGradient)
