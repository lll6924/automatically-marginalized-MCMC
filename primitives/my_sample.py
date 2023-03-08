from numpyro import sample
from jax import core
from jax._src import abstract_arrays
from jax.interpreters import ad, xla
from jax._src.lib import xla_client

register_p = core.Primitive("register")

def register(x):
    return x

def register_abstract_eval(x):
  return abstract_arrays.ShapedArray(x.shape, x.dtype)

def register_xla_translation(ctx, avals_in, avals_out, x):
  return [x]

def register_value_and_jvp(arg_values, arg_tangents):
    return arg_values[0], arg_tangents[0]

register_p.def_impl(register)
register_p.def_abstract_eval(register_abstract_eval)
ad.primitive_jvps[register_p] = register_value_and_jvp
xla.register_translation(register_p, register_xla_translation, platform='cpu')

def my_sample(*args, **kwargs):
    variable = sample(*args, **kwargs)
    x = register_p.bind(variable)
    return x