from dataset.electric import *

from jax import numpy as jnp
import numpyro.distributions as dist
from primitives import my_sample
import numpyro
from numpyro.infer.reparam import TransformReparam
from model import Model

class ElectricCompanyReparameterized(Model):

    """
        Electric company model with reparameterization implemented with scalars
    """

    def __init__(self):
        self.N = N
        self.n_pair = n_pair
        self.n_grade = n_grade
        self.n_grade_pair = n_grade_pair
        self.grade = grade
        self.grade_pair = grade_pair
        self.pair = pair
        self.treatment = jnp.array(treatment)
        self.y = jnp.array(y)


    def model(self,y = None):
        if y is None:
            y = [None for _ in range(self.N)]
        muas = []
        for i in range(self.n_grade_pair):
            mua = my_sample('mua{}'.format(str(i)), dist.Normal(0., 1.))
            muas.append(mua)

        sigmays = []
        for i in range(self.n_grade):
            sigmay = my_sample('sigmay{}'.format(str(i)), dist.Normal(0., 1.))
            sigmays.append(sigmay)

        aa = []
        for i in range(self.n_pair):
            with numpyro.handlers.reparam(config={'a{}'.format(str(i)): TransformReparam()}):
                a = numpyro.sample(
                    'a{}'.format(str(i)),
                    dist.TransformedDistribution(dist.Normal(0., 1.),
                                                 dist.transforms.AffineTransform(100. * muas[self.grade_pair[i]-1], 1.)))
                aa.append(a)

        bs = []
        for i in range(self.n_grade):
            b = my_sample('b{}'.format(str(i)), dist.Normal(0., 100.))
            bs.append(b)

        for i in range(self.N):
            mean = aa[self.pair[i]-1]+bs[self.grade[i]-1]*self.treatment[i]
            std = jnp.exp(sigmays[self.grade[i]-1])
            obs = my_sample('y{}'.format(str(i)), dist.Normal(mean, std), obs= y[i])


    def args(self):
        return ()

    def kwargs(self):
        return {'y':self.y}

    def name(self):
        return 'ElectricCompany'