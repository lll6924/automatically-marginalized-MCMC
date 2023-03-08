import jax.numpy as jnp

def register(result, arguments, params):
    def fun(*args, **kwargs):
        if result[0].name in kwargs['stored'].keys():
            return kwargs['stored'][result[0].name], kwargs['stored']
        result[0].result, kwargs['stored'] = arguments[0].value(*args, **kwargs)
        kwargs['stored'][result[0].name] = result[0].result
        return result[0].result, kwargs['stored']
    result[0].value = fun

def unary_rule_helper(result, arguments, params, f):
    assert (len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]

    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = f(i1_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def tan(result, arguments, params):
    return unary_rule_helper(result, arguments, params, jnp.tan)

def abs(result, arguments, params):
    return unary_rule_helper(result, arguments, params, jnp.abs)


def inv(result, arguments, params):
    return unary_rule_helper(result, arguments, params, lambda x: 1./x)


def square(result, arguments, params):
    return unary_rule_helper(result, arguments, params, jnp.square)


def sqrt(result, arguments, params):
    return unary_rule_helper(result, arguments, params, jnp.sqrt)


