import jax.lax

def sub(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 2)
    i1 = arguments[0]
    i2 = arguments[1]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = i1_val - i2_val
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def mul(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 2)
    i1 = arguments[0]
    i2 = arguments[1]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = i1_val * i2_val
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

def add(result, arguments, params):
    assert(len(result) == 1 and len(arguments) == 2)
    i1 = arguments[0]
    i2 = arguments[1]
    o = result[0]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = i1_val + i2_val
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun


def max(result, arguments, params):
    assert (len(result) == 1)
    o = result[0]
    i1 = arguments[0]
    i2 = arguments[1]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = jax.lax.max(i1_val,i2_val)
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun


def div(result, arguments, params):
    assert (len(result) == 1)
    o = result[0]
    i1 = arguments[0]
    i2 = arguments[1]
    def fun(*args, **kwargs):
        if o.name in kwargs['stored'].keys():
            return kwargs['stored'][o.name], kwargs['stored']
        i1_val, kwargs['stored'] = i1.value(*args, **kwargs)
        i2_val, kwargs['stored'] = i2.value(*args, **kwargs)
        o.result = i1_val / i2_val
        kwargs['stored'][o.name] = o.result
        return o.result, kwargs['stored']
    o.value = fun

