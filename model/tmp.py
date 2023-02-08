import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from primitives import my_sample
from model import Model

class TMP(Model):
    """
        The example model in Appendix F
    """
    def __init__(self, N=100):
        self.N = int(N)
    def model(self, y = None):
        if y is None:
            y = [None for _ in range(self.N)]
        x = my_sample('x', dist.Normal(0., 1.))
        logsigma = my_sample('sigma', dist.Normal(0., 1.))
        sigma = jnp.exp(logsigma)
        for i in range(self.N):
            obs = my_sample('y{}'.format(i), dist.Normal(x , sigma), obs = y[i])


    def args(self):
        return ()

    def kwargs(self):
        return {'y':jnp.zeros(self.N)}

    def name(self):
        return 'TMP'
