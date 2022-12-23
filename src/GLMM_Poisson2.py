import numpyro.distributions as dist

from jax.lax import fori_loop, dynamic_update_index_in_dim 
from jax.numpy import array, zeros, ones, empty
from jax import jit, vmap

#no further changes
def convert_inputs(inputs:dict):
    nsite = inputs['nsite']
    nyear = inputs['nyear']
    # supposed to be a matrix
    C = array(inputs['C'], dtype=dtype_long)
    C = C.reshape((nyear, nsite))
    year = array(inputs['year'], dtype=dtype_float)
    # assertions to ensure the proper parameters
    assert(nsite >= 0)
    assert(nyear >= 0)
    return { 'nsite': nsite, 'nyear': nyear, 'C': C, 'year': year }

#no further changes
def transformed_data(*, C, year):
    # Transformed data
    year_squared = year * year
    year_cubed = year * year * year
    return { 'year_squared': year_squared, 'year_cubed': year_cubed }

def model(*, nsite, nyear, C, year, year_squared, year_cubed):
    # Parameters
    mu = sample('mu', dist.improper_uniform(shape=[]))
    alpha = sample('alpha', dist.improper_uniform(shape=[nsite]))
    eps = sample('eps', dist.improper_uniform(shape=[nyear]))
    beta__ = sample('beta', dist.improper_uniform(shape=[3]))
    sd_alpha = sample('sd_alpha', dist.uniform(0, 2))
    sd_year = sample('sd_year', dist.uniform(0, 1))
    # Transformed parameters
    log_lambda = empty([nyear, nsite], dtype=dtype_float)

    @jit
    def _fori__1(i, _acc__2):
        log_lambda = _acc__2
        log_lambda = dynamic_update_index_in_dim(log_lambda,  alpha + beta__[
        1 - 1] * year[i - 1] + beta__[2 - 1] * year_squared[i - 1] + beta__[
        3 - 1] * year_cubed[i - 1] + eps[i - 1], ops_index[i - 1], 0)
        return log_lambda
    
    log_lambda = fori_loop(1, nyear + 1, _fori__1, log_lambda)
    # Model
    observe('_alpha__3', dist.normal(mu, sd_alpha), alpha)
    observe('_mu__4', dist.normal(0, 10), mu)
    observe('_beta__5', dist.normal(0, 10), beta__)
    observe('_eps__6', dist.normal(0, sd_year), eps)
    def _fori__7(i, _acc__8):
        observe(f'_C__{i}__9', dist.poisson_log(log_lambda[i - 1]), C[i - 1])
        return None
    _ = fori_loop(1, nyear + 1, _fori__7, None)


def generated_quantities(*, nsite, nyear, C, year, year_squared, year_cubed,
                            mu, alpha, eps, beta__, sd_alpha, sd_year):
    # Transformed parameters
    log_lambda = empty([nyear, nsite], dtype=dtype_float)
    @jit
    def _fori__10(i, _acc__11):
        log_lambda = _acc__11
        log_lambda = dynamic_update_index_in_dim(log_lambda, alpha + beta__[
        1 - 1] * year[i - 1] + beta__[2 - 1] * year_squared[i - 1] + beta__[
        3 - 1] * year_cubed[i - 1] + eps[i - 1], ops_index[i - 1], 0)
        return log_lambda
    log_lambda = fori_loop(1, nyear + 1, _fori__10, log_lambda)
    return { 'log_lambda': log_lambda }

def map_generated_quantities(_samples, *, nsite, nyear, C, year,
                                          year_squared, year_cubed):
    def _generated_quantities(mu, alpha, eps, beta__, sd_alpha, sd_year):
        return generated_quantities(nsite=nsite, nyear=nyear, C=C, year=year,
                                    year_squared=year_squared,
                                    year_cubed=year_cubed, mu=mu,
                                    alpha=alpha, eps=eps, beta__=beta__,
                                    sd_alpha=sd_alpha, sd_year=sd_year)
    _f = jit(vmap(_generated_quantities))
    return _f(_samples['mu'], _samples['alpha'], _samples['eps'],
              _samples['beta'], _samples['sd_alpha'], _samples['sd_year'])
