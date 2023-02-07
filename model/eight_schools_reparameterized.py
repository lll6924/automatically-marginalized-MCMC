import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from primitives import my_sample
from numpyro.infer.reparam import TransformReparam

class EightSchoolsReparameterized:
    J = 8
    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    # Eight Schools example with reparameterization
    def model(self, sigma, obs = None):
        mu = my_sample('mu', dist.Normal(0, 5))
        tau = my_sample('tau', dist.HalfCauchy(5))
        with numpyro.plate('J', self.J):
            with numpyro.handlers.reparam(config={'theta': TransformReparam()}):
                theta = numpyro.sample(
                    'theta',
                    dist.TransformedDistribution(dist.Normal(0., 1.),
                                             dist.transforms.AffineTransform(mu, tau)))
            obs = my_sample('obs', dist.Normal(theta, sigma), obs=obs)
        return obs

    def args(self):
        return (self.sigma,)

    def kwargs(self):
        return {'obs':self.y}

    def name(self):
        return 'EightSchools'
