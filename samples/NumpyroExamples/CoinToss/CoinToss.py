from numpyro import sample
import numpyro.distributions as dist

# Simple Coin Toss example
def coin_toss(c1, c2):
    """
    A simple model of tossing two coins independently.
    The 'measurements' (observed outcomes) are stored as arrays
    of {0 = Head,1 = Tail} in the JSON file.
    The goal is to determine the coin bias (p1, p2).

    **Assumption**: No a-priori knowledge regarding the bias of either coin.
    **Conclusion**: Model for either p is a uniform distribution on [0,1]

    The observed results `obs1, obs2` follow a Bernoulli distribution ([0,1] ->{0,1})
    and are given the observed results as `obs` arguments.
    """
    theta1 = sample("p1", dist.Uniform(0, 1))
    theta2 = sample("p2", dist.Uniform(0, 1))

    sample("obs1", dist.Bernoulli(theta1), obs=c1)
    sample("obs2", dist.Bernoulli(theta2), obs=c2)
