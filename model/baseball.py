import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import BASEBALL, load_dataset
import jax.numpy as jnp
from primitives import my_sample
from model import Model


class Baseball(Model):

    """
        Partially pooled model for Baseball problem
        Original example: https://pyro.ai/numpyro/examples/baseball.html
    """


    def __init__(self):
        _, fetch_train = load_dataset(BASEBALL, split='train', shuffle=False)
        train, player_names = fetch_train()
        _, fetch_test = load_dataset(BASEBALL, split='test', shuffle=False)
        test, _ = fetch_test()
        self.at_bats, self.y = train[:, 0], train[:, 1]
        self.season_at_bats, self.season_hits = test[:, 0], test[:, 1]


    def model(self, at_bats, hits=None):
        m = my_sample("m", dist.Uniform(0, 1))
        kappa = my_sample("kappa", dist.Pareto(1., 1.5))
        num_players = at_bats.shape[0]

        with numpyro.plate("num_players", num_players):
            phi_prior = dist.Beta(m * kappa, (1 - m) * kappa)
            phi = my_sample("phi", phi_prior)
            return my_sample("obs", dist.Binomial(at_bats, probs=phi), obs=hits)

    def args(self):
        return (self.at_bats,)


    def kwargs(self):
        return {'hits' : self.y}


    def name(self):
        return 'Baseball'
