import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.profiler.annotate_function, name="body_func")
def body_func(i, val):
	with jax.profiler.StepTraceAnnotation("LF-body_step", step_num=i):
		val *= (i+1)
		return val


def factorial(n):
	output = jax.lax.fori_loop(0, n, body_func, 1)
	print(f'{n}!={output}')
	return output

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
	factorial(20)