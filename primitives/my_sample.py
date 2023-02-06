from numpyro import sample

from jax import jit

@jit
def register(v):
    return v

def my_sample(*args, **kwargs):
    variable = sample(*args, **kwargs)
    return register(variable)