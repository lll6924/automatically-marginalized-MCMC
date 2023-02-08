import jax.lax


def binary_rule_helper(result, arguments, params, f):
    assert (len(result) == 1 and len(arguments) == 2)
    i1 = arguments[0]
    i2 = arguments[1]
    o = result[0]

    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = f(i1_val, i2_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']

    o.value = fun


def sub(result, arguments, params):
    return binary_rule_helper(result, arguments, params, lambda x, y: x - y)

def mul(result, arguments, params):
    return binary_rule_helper(result, arguments, params, lambda x, y: x * y)

def add(result, arguments, params):
    return binary_rule_helper(result, arguments, params, lambda x, y: x + y)

def max(result, arguments, params):
    return binary_rule_helper(result, arguments, params, jax.lax.max)

def div(result, arguments, params):
    return binary_rule_helper(result, arguments, params, lambda x, y: x / y)


