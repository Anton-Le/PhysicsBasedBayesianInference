from numpyro import sample
import numpyro.distributions as dist

def timeShift(t):
    """
    Function to shift the time s.t. the first element of the time
    array is always 0.
    """
    t0 = t[0]
    return t[:] - t0


# Simple linear acceleration example
def linear_accel(t, z, sigmaObs):
    """
    This model is a simple model of a measurement of 
    the position of an arrow shot vertically up
    from a height h with initial velocity v0
    subject to the earths' acceleration g.

    The goal is to determine h, v0, g from the measurement
    data (t_i, z_i) assuming measurement uncertainty of sigmaObs.
    """
    t_shifted = timeShift(t)
    h = sample("h", dist.Uniform(70, 150) )
    v0 = sample("v0", dist.Uniform(20, 100) )
    g = sample("g", dist.Normal(-5, 2) )

    sample("z", dist.Normal(h + v0*t_shifted + g*t_shifted**2, sigmaObs), obs=z)
