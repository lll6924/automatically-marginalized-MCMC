import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from primitives import my_sample

class GammaModel:
    def model(self, x3 = None, x4 = None):
        x1 = my_sample('x1', dist.Normal(0, 5))
        x2 = my_sample('x2', dist.Gamma(jnp.exp(x1), 1.))
        x3 = my_sample('x3', dist.Gamma(jnp.exp(x1), 2*x2), obs = x3)
        x4 = my_sample('x4', dist.Exponential(x2), obs = x4)

    def args(self):
        return ()

    def kwargs(self):
        return {'x3':jnp.array(2.), 'x4':jnp.array(5.)}

    def name(self):
        return 'GammaModel'
