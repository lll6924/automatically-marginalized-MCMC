import jax.numpy as jnp
import click
import ast

def log_mean_exp(x, axis=None):
    m = jnp.max(x, axis=axis)
    return m + jnp.log(jnp.mean(jnp.exp(x-m), axis=axis))

class PythonLiteralOption(click.Option):
    # codes from https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)