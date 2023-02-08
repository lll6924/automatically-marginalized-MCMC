import numpyro.distributions as dist
from utils import get_last_alphabetic
import jax.numpy as jnp
import jax.scipy as jsc
from rule import Variable

"""
    This file defines the distribution families in this project as classes.
    For each family, a function `get_parameters` is used for extracting its parameter variables from Jaxprs.
    The extraction of parameters is sensitive to the version of JAX and NumPyro. Make sure you use JAX==0.4.1 and NumPyro=0.10.1.
"""

class MyNormal:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        if pred[rv['expr']][0] == lst:
            mean = pred[rv['expr']][1]
        else:
            mean = pred[rv['expr']][0]
        lst_lst = get_last_alphabetic(lst)
        if pred[lst][0] == lst_lst:
            std = pred[lst][1]
        else:
            std = pred[lst][0]
        return mean, std
    @staticmethod
    def evaluate(params, value):
        mean, std = params
        d = dist.Normal(mean, std)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        mean, std = params
        d = dist.Normal(mean, std)
        return d.sample(rng_key)


class MyHalfCauchy:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        lst_lst = get_last_alphabetic(lst)
        lst_lst_lst = get_last_alphabetic(lst_lst)
        if pred[lst_lst][0] == lst_lst_lst:
            scale = pred[lst_lst][1]
        else:
            scale = pred[lst_lst][0]
        return (scale,)
    @staticmethod
    def evaluate(params, value):
        scale = params[0]
        d = dist.HalfCauchy(scale)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        scale = params[0]
        d = dist.HalfCauchy(scale)
        return d.sample(rng_key)

class MyGamma:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        if pred[rv['expr']][0] == lst:
            beta = pred[rv['expr']][1]
        else:
            beta = pred[rv['expr']][0]
        lst_lst = get_last_alphabetic(lst)
        if pred[lst][0] == lst_lst:
            alpha = pred[lst][1]
        else:
            alpha = pred[lst][0]
        return alpha, beta
    @staticmethod
    def evaluate(params, value):
        alpha, beta = params
        d = dist.Gamma(alpha, beta)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        alpha, beta = params
        d = dist.Gamma(alpha, beta)
        return d.sample(rng_key)


class MyExponential:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        if pred[rv['expr']][0] == lst:
            lamb = pred[rv['expr']][1]
        else:
            lamb = pred[rv['expr']][0]
        return (lamb,)
    @staticmethod
    def evaluate(params, value):
        lamb = params[0]
        d = dist.Exponential(lamb)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        lamb = params[0]
        d = dist.Exponential(lamb)
        return d.sample(rng_key)

class MyCompoundGamma:
    @staticmethod
    def get_parameters(pred, rv):
        raise NotImplementedError()
    @staticmethod
    def evaluate(params, value):
        alpha, beta, q = params
        base = jnp.log(value/q) * (alpha - 1) + jnp.log(1 + value/q) * (- alpha - beta) - jnp.log(q)
        div = jsc.special.gammaln(alpha) + jsc.special.gammaln(beta) - jsc.special.gammaln(alpha + beta)
        return base - div

    @staticmethod
    def sample(params, rng_key):
        raise NotImplementedError()

class MyBeta:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        if pred[rv['expr']][0] == lst:
            lst1 = pred[rv['expr']][1]
        else:
            lst1 = pred[rv['expr']][0]
        lst2 = pred[lst1][0]

        lst2_lst = get_last_alphabetic(lst2)
        if pred[lst2][0] == lst2_lst:
            lst3 = pred[lst2][1]
        else:
            lst3 = pred[lst2][0]
        lst4 = pred[lst3][0]
        ##lst4_lst = get_last_alphabetic(lst4)
        lst5 = pred[lst4][0]
        lst6 =pred[lst5][1]
        # if pred[lst4][0] == lst4_lst:
        #     lst5 = pred[lst4][1]
        # else:
        #     lst5 = pred[lst4][0]
        # lst5_lst = get_last_alphabetic(lst5)
        # lst6 = pred[lst5_lst][1]
        if Variable.all[lst6].operator == 'convert_element_type':
            lst6 = pred[lst6][0]
        if Variable.all[lst6].operator == 'broadcast_in_dim':
            lst6 = pred[lst6][0]
        beta = get_last_alphabetic(lst6)
        while Variable.all[beta].operator != 'broadcast_in_dim':
            beta = get_last_alphabetic(beta)
        alpha = get_last_alphabetic(beta)
        while Variable.all[alpha].operator != 'broadcast_in_dim':
            alpha = get_last_alphabetic(alpha)
        return pred[alpha][0], pred[beta][0]
    @staticmethod
    def evaluate(params, value):
        alpha, beta = params
        d = dist.Beta(alpha, beta)
        return d.log_prob(value)
    @staticmethod
    def sample(params, rng_key):
        alpha, beta = params
        d = dist.Beta(alpha, beta)
        return d.sample(rng_key)

class MyBernoulli:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        return (pred[lst][1],)
    @staticmethod
    def evaluate(params, value):
        lamb = params[0]
        d = dist.BernoulliProbs(lamb)
        return d.log_prob(value)
    @staticmethod
    def sample(params, rng_key):
        lamb = params[0]
        d = dist.BernoulliProbs(lamb)
        return d.sample(rng_key)

class MyBinomial:
    @staticmethod
    def get_parameters(pred, rv):
        return pred[rv['expr']][5], pred[rv['expr']][6]
    @staticmethod
    def evaluate(params, value):
        lamb, cnt = params
        d = dist.BinomialProbs(lamb, cnt)
        return d.log_prob(value)
    @staticmethod
    def sample(params, rng_key):
        lamb, cnt = params
        d = dist.BinomialProbs(lamb, cnt)
        return d.sample(rng_key)

class MyBetaBinomial:
    @staticmethod
    def get_parameters(pred, rv):
        raise NotImplementedError()
    @staticmethod
    def evaluate(params, value):
        alpha, beta, cnt = params
        d = dist.BetaBinomial(alpha, beta, cnt)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        raise NotImplementedError()


class MyUniform:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        for _ in range(3):
            lst = get_last_alphabetic(lst)

        return pred[lst][1], pred[lst][0]
    @staticmethod
    def evaluate(params, value):
        alpha, beta = params
        d = dist.Uniform(alpha, beta)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        alpha, beta = params
        d = dist.Uniform(alpha, beta)
        return d.sample(rng_key)

class MyPareto:
    @staticmethod
    def get_parameters(pred, rv):
        lst = get_last_alphabetic(rv['expr'])
        lst_lst = get_last_alphabetic(lst)
        if pred[lst][0] == lst_lst:
            alpha = pred[lst][1]
        else:
            alpha = pred[lst][0]
        lst3 = get_last_alphabetic(lst_lst)
        lst3_lst = get_last_alphabetic(lst3)
        if pred[lst3][0] == lst3_lst:
            beta = pred[lst3][1]
        else:
            beta = pred[lst3][0]
        return alpha, beta
    @staticmethod
    def evaluate(params, value):
        alpha, beta = params
        d = dist.Pareto(alpha, beta)
        return d.log_prob(value)

    @staticmethod
    def sample(params, rng_key):
        alpha, beta = params
        d = dist.Pareto(alpha, beta)
        return d.sample(rng_key)

distribution_mapping = {
    'Normal': MyNormal,
    'HalfCauchy': MyHalfCauchy,
    'Gamma': MyGamma,
    'Exponential': MyExponential,
    'CompoundGamma': MyCompoundGamma,
    'Beta': MyBeta,
    'BernoulliProbs': MyBernoulli,
    'BinomialProbs': MyBinomial,
    'BetaBinomial': MyBetaBinomial,
    'Uniform': MyUniform,
    'Pareto': MyPareto
}

if __name__ == '__main__':
    x1 = jnp.array(4.)
    x2 = jnp.array(5.)
    print(MyGamma.evaluate(2., 3., x1), MyGamma.evaluate(6., x1, x2))
    print(MyGamma.evaluate(8., 3.+x2, x1), MyCompoundGamma.evaluate(6.,2.,3., x2))
