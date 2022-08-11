import jax.numpy as jnp
from jax import pmap, vmap
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2' 
# apparently this can be artificially set to any number. I'm not sure of the consequences though.

def propagate_particle(initial_position, velocity, time):
	'''
	Return position of particle after time.
	'''
	return initial_position + velocity * time

initial_position = jnp.array([1., 2.])
velocity = jnp.array([-2., -1.])
time = jnp.array([1.])
final_position = propagate_particle(initial_position, velocity, time)

print('\n')
print('Serial:')
print(f'{initial_position=}')
print(f'{velocity=}')
print(f'{time=}')
print(f'{final_position=}')
print('#' * 45 + '\n')

initial_positions = jnp.array([
	[1., 2.], 
	[2., 4.]
	])
velocitys = jnp.array([
	[-2., -1.],
	[-1., -2],
	])
times = jnp.array([1, 2])

# https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap

pmap_propagate_particle = pmap(
	propagate_particle,
	in_axes=0 # specifies which axis to map over - LEN(AXIS 0) must be <= # devices
	)


final_positions = pmap_propagate_particle(initial_positions, velocitys, times) 



print('pmap:')
print(f'{initial_positions=}')
print(f'{velocitys=}')
print(f'{times=}')
print(f'{final_positions=}')
print('#' * 45 + '\n')

initial_positions = jnp.array([
	[1., 2.], 
	[2., 4.],
	[1., 2.], 
	[2., 4.],
	[1., 2.], 
	[2., 4.],
	])
velocitys = jnp.array([
	[-2., -1.],
	[-1., -2],
	[-2., -1.],
	[-1., -2],
	[-2., -1.],
	[-1., -2],
	])
times = jnp.array([1, 2, 1, 2, 1, 2])

vmap_propagate_particle = vmap(
	propagate_particle,
	in_axes=0 # specifies which axis to map over - NO LIMIT ON LEN(AXIS 0) 
	)

final_positions = vmap_propagate_particle(initial_positions, velocitys, times) 

print('vmap:')
print(f'{initial_positions=}')
print(f'{velocitys=}')
print(f'{times=}')
print(f'{final_positions=}')
print('#' * 45 + '\n')