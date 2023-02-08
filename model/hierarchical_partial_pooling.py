import numpyro
import numpyro.distributions as dist
from dataset.repeated_binary_trials import get_repeated_binary_trials_data
from primitives import my_sample
from model import Model

class HierarchicalPartialPooling(Model):

    """
        Hierarchical partial pooling model for binary repeated trials
        See https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html
    """

    def __init__(self, dataset = 'baseball_small'):
        """
            dataset: the name of the dataset. Choose from ['baseball_small', 'baseball_large', 'rat_tumors']
        """
        self.x, self.y, self.test_x, self.test_y = get_repeated_binary_trials_data(dataset)
        self.N = len(self.x)

    def model(self, x, y=None):
        m = my_sample("m", dist.Uniform(0, 1))
        kappa = my_sample("kappa", dist.Pareto(1, 1.5))

        with numpyro.plate("N", self.N):
            phi_prior = dist.Beta(m * kappa, (1 - m) * kappa)
            phi = my_sample("phi", phi_prior)
            return my_sample("obs", dist.Binomial(x, probs=phi), obs=y)

    def args(self):
        return (self.x,)

    def kwargs(self):
        return {'y':self.y}

    def name(self):
        return 'HierarchialPartialPooling'