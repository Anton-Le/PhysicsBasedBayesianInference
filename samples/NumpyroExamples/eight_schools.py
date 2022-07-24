from numpyro import sample, plate
import numpyro.distributions as dist

# Eight Schools example
def eight_schools(J, sigma, y):
    mu = sample('mu', dist.Normal(0, 5))
    tau = sample('tau', dist.HalfCauchy(5))
    with plate('J', J):
        theta = sample('theta', dist.Normal(mu, tau))
        sample('obs', dist.Normal(theta, sigma), obs=y)