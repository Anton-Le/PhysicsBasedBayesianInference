#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:24:44 2022

@author: thomas
"""



import jax.numpy as jnp
from jax import grad, vmap, jit
import jax
from mpi4py import MPI, rc
import mpi4jax
rc.threaded = True
rc.thread_level = "funneled"
comm = MPI.COMM_WORLD
rank = comm.Get_rank ()
size = comm.Get_size ()

# jax.config.omnistaging_enabled = True

def HMC_MPI(hmc, numIterations, mass, seed):
	'''
	@parameters
		hmc (HMC object):
		numIterations (int): Number of HMC steps
		mass (ndarray): array of masses to make up ensemble
		seed (int): Seed for RNG
	'''
	if rank == 0:
		key = jax.random.PRNGKey(seed)
		keys_0 = jax.random.split(key, num=len(mass)*size)
		keys_0 = jnp.reshape(keys_0, newshape=(size, len(mass), 2))
		print(keys_0.shape)
		
	else:
		keys_0 = jnp.zeros((size, len(mass), 2), dtype='uint32')
		print(keys_0.shape)
	# print(f'{mass=}')
	# jeff = mpi4jax.bcast(mass, root=0, comm=COMM)

	keys, _ = mpi4jax.scatter(keys_0, root=0, comm=comm)
	print(f'{keys.shape=}')
	# print(keys.shape)
	# print(f'{rank}: {keys=}')
	# samples, _ = hmc.getSamples(numIterations, mass, keys)
	# samples, _token = mpi4jax.gather(samples, root=0, comm=comm)

	# if rank == 0:
	# 	return samples
	# return None



# I have come across a slight bug which is easily fixed.
# It appears that mpi4jax checks that jax.config.omnistaging_enabled=True. If it isn't true I get this (long) error.

# <details>
#   <summary>Error</summary>
#   ```
# Traceback (most recent call last):
#   File "test_HMC_MPI.py", line 64, in <module>
#     init()
#   File "test_HMC_MPI.py", line 60, in init
#     samples = HMC_MPI(hmcObject, numIterations, mass, seed)
#   File "/home/tom/Dev/PhysicsBasedBayesianInference/src/tests/../HMC_MPI.py", line 37, in HMC_MPI
#     keys, _token = mpi4jax.scatter(keys, 0)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/validation.py", line 90, in wrapped
#     return function(*args, **kwargs)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/collective_ops/scatter.py", line 86, in scatter
#     mpi_scatter_p.bind(
# jax._src.source_info_util.JaxStackTraceBeforeTransformation: AttributeError: 'Config' object has no attribute 'omnistaging_enabled'

# The preceding stack trace is the source of the JAX operation that, once transformed by JAX, triggered the following exception.

# --------------------

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "test_HMC_MPI.py", line 64, in <module>
#     init()
#   File "test_HMC_MPI.py", line 60, in init
#     samples = HMC_MPI(hmcObject, numIterations, mass, seed)
#   File "/home/tom/Dev/PhysicsBasedBayesianInference/src/tests/../HMC_MPI.py", line 37, in HMC_MPI
#     keys, _token = mpi4jax.scatter(keys, 0)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/validation.py", line 90, in wrapped
#     return function(*args, **kwargs)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/collective_ops/scatter.py", line 86, in scatter
#     mpi_scatter_p.bind(
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/core.py", line 324, in bind
#     return self.bind_with_trace(find_top_trace(args), args, params)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/core.py", line 327, in bind_with_trace
#     out = trace.process_primitive(self, map(trace.full_raise, args), params)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/core.py", line 684, in process_primitive
#     return primitive.impl(*tracers, **params)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/dispatch.py", line 99, in apply_primitive
#     compiled_fun = xla_primitive_callable(prim, *unsafe_map(arg_spec, args),
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/util.py", line 220, in wrapper
#     return cached(config._trace_context(), *args, **kwargs)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/util.py", line 213, in cached
#     return f(*args, **kwargs)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/dispatch.py", line 164, in xla_primitive_callable
#     compiled = _xla_callable_uncached(lu.wrap_init(prim_fun), device, None,
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/dispatch.py", line 248, in _xla_callable_uncached
#     return lower_xla_callable(fun, device, backend, name, donated_invars, False,
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/profiler.py", line 294, in wrapper
#     return func(*args, **kwargs)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/_src/dispatch.py", line 376, in lower_xla_callable
#     lowering_result = mlir.lower_jaxpr_to_module(
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 609, in lower_jaxpr_to_module
#     lower_jaxpr_to_fun(
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 868, in lower_jaxpr_to_fun
#     out_vals, tokens_out = jaxpr_subcomp(ctx.replace(name_stack=callee_name_stack),
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 995, in jaxpr_subcomp
#     ans = rule(rule_ctx, *map(_unwrap_singleton_ir_values, in_nodes),
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 1248, in cached_lowering
#     func = _emit_lowering_rule_as_fun(partial(f, **params), ctx)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 917, in _emit_lowering_rule_as_fun
#     outs = lowering_rule(sub_ctx, *_unwrap_singleton_ir_values(unflattened_args))
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/mlir.py", line 1304, in fallback
#     xla_computation = xla.primitive_subcomputation(
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/xla.py", line 311, in primitive_subcomputation
#     ans = rule(ctx, avals_in, avals_out, *xla_args, **params)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/jax/interpreters/xla.py", line 543, in wrapped
#     ans = f(ctx.builder, *args, **kw)
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/decorators.py", line 95, in wrapped
#     f()
#   File "/programs/extension/tmp/NumPyroEnv/lib/python3.8/site-packages/mpi4jax/_src/decorators.py", line 30, in ensure_omnistaging
#     if not jax.config.omnistaging_enabled:
# AttributeError: 'Config' object has no attribute 'omnistaging_enabled'
#   ```
# </details>
# The jax I gave installed doesn't have jax.config.omnistaging_enabled, but putting jax.config.omnistaging_enabled at the top of the file seems to fix this.

