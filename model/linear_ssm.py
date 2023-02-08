import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from primitives import my_sample
from model import Model

class LinearSSM(Model):
    """
        A simple linear state space model
    """

    y1 = jnp.array(1.)
    y2 = jnp.array(2.)
    y3 = jnp.array(3.)

    def model(self, y1 = None, y2 = None, y3 = None):
        mu = my_sample('mu', dist.Normal(0, 5))
        tau = my_sample('tau', dist.HalfCauchy(5))
        x1 = my_sample('x1', dist.Normal(mu, tau))
        y1 = my_sample('y1', dist.Normal(x1,1.), obs=y1)
        x2 = my_sample('x2', dist.Normal(x1, tau))
        y2 = my_sample('y2', dist.Normal(x2,1.), obs=y2)
        x3 = my_sample('x3', dist.Normal(x2, tau))
        y3 = my_sample('y3', dist.Normal(x3,1.), obs=y3)

    def args(self):
        return ()

    def kwargs(self):
        return {'y1':self.y1,'y2':self.y2,'y3':self.y3}

    def name(self):
        return 'LinearSSM'
