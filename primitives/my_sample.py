from numpyro import sample

from jax import jit

@jit
def register(v):
    """
        We use this function as a hack to annotate random variables in Jaxprs
    """
    return v

def my_sample(*args, **kwargs):
    variable = sample(*args, **kwargs)
    return register(variable)