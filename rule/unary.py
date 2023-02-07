import jax.numpy as jnp

def xla_call(result, arguments, params): # other xla_calls are already dealt with in variables.update_callable
    assert (params['name'] == 'register')
    def fun(*args, **kwargs):
        if result[0].name in kwargs['stored'].keys():
            return kwargs['stored'][result[0].name], kwargs['stored']
        result[0].result, kwargs['stored'] = arguments[0].value(*args, **kwargs)
        kwargs['stored'][result[0].name] = result[0].result
        return result[0].result, kwargs['stored']
    result[0].value = fun

def tan(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = jnp.tan(i1_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def abs(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = jnp.abs(i1_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def inv(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = 1/i1_val
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def square(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = jnp.square(i1_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def sqrt(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 1)
    i = arguments[0]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i.value(*args, **kwargs)
        o.result = jnp.sqrt(i1_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

