import jax.core
import jax.numpy as jnp
import rule
import random
import string
from functools import partial

def zero_fun(*args, **kwargs):
    return 0., kwargs['stored']

def one_fun(*args, **kwargs):
    return 1., kwargs['stored']

def random_name():
    name = ''.join(random.choice(string.digits) for _ in range(10))
    while name in Variable.all.keys():
        name = ''.join(random.choice(string.digits) for _ in range(10))
    return name

class Variable:
    idCounter = 0
    all = {}
    def __init__(self,
                 name = None,
                 value = None,
                 determinstic = True,
                 parents = None,
                 dependency = None
                 ):
        if name is None:
            name = random_name()
        self.name = name
        Variable.all[name] = self
        self.is_determinstic = determinstic
        self.operator = None
        self.operands = []
        self.params = None
        Variable.idCounter += 1
        self.id = Variable.idCounter
        def fun(*args, **kwargs):
            return value, kwargs['stored']
        self.value = fun
        self.constant = False
        if value is None:
            self.nonlinear = zero_fun
        else:
            self.constant = True
            self.nonlinear = fun
        self.trigger = fun
        self.result = value

        def substitute(*args, **kwargs):
            self.result = kwargs['substitute'][name]
            return self.result, kwargs['stored']
        self.substitute = substitute
        self.parameters = tuple()

    def get_parameters(self, substitute, stored):
        tup = tuple()
        for e in self.parameters:
            v, stored = e.value(substitute=substitute, stored = stored)
            tup = tup + (v,)
        return tup, stored



def operator_map(operator):
    if operator in ['or']:
        return operator + '_'
    return operator

def update_callable(result, arguments, eqn):
    """
        Define a new computation graph based on an operation over current computation graph
    """
    ins = [a.value for a in arguments]
    def fun(index, *args, **kwargs):
        if result[index].name in kwargs['stored'].keys():
            return kwargs['stored'][result[index].name], kwargs['stored']
        invals = []
        for i in ins:
            v, kwargs['stored'] = i(*args, **kwargs)
            invals.append(v)
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        kwargs['stored'][result[index].name] = outvals[index]
        return outvals[index], kwargs['stored']
    for i in range(len(result)):
        result[i].value = partial(fun,i)

def update(result, operator, arguments, params = None, eqn = None):
    if eqn is None:
        fun = getattr(rule, operator_map(operator))
        fun(result, arguments, params)
    else:
        params = eqn.params
        update_callable(result, arguments, eqn)
    for r in result:
        r.operator = operator
        r.operands = arguments
        r.params = params
        if not r.is_determinstic:
            r.trigger = r.value
            r.value = r.substitute


def is_linear(v1, v2, stored):
    """
        Linearity detection
        Return: (whether the coefficient is positive, whether the intercept is positive, whether v1 is linear w.r.t. v2), stored
        stored: a dictionary to store intermediate results with the same v2. This trick is crucial for large models.
    """
    if v1.name in stored.keys():
        return stored[v1.name], stored
    if v1 == v2:
        return (True, False, True), stored
    if not v1.is_determinstic or v1.constant:
        return (False, True, True), stored
    if v1.operator == 'add' or v1.operator == 'sub':
        (a, b, p), stored = is_linear(v1.operands[0], v2, stored)
        (c, d, q), stored = is_linear(v1.operands[1], v2, stored)
        stored[v1.name] = (a or c, b or d, p and q)
        return stored[v1.name], stored

    if v1.operator == 'mul':
        (a, b, p), stored = is_linear(v1.operands[0], v2, stored)
        (c, d, q), stored = is_linear(v1.operands[1], v2, stored)
        if not a:
            stored[v1.name] = (b and c, b and d, p and q)
        elif not c:
            stored[v1.name] = (d and a, d and b, p and q)
        else:
            stored[v1.name] = (False, False, False)
        return stored[v1.name], stored

    if v1.operator == 'div':
        (a, b, p), stored = is_linear(v1.operands[0], v2, stored)
        (c, d, q), stored = is_linear(v1.operands[1], v2, stored)
        if not c:
            stored[v1.name] = (a, b, p and q)
        else:
            stored[v1.name] = (False, False, False)
        return stored[v1.name], stored

    if v1.operator == 'register':
        return is_linear(v1.operands[0], v2, stored)

    for v in v1.operands:
        (a, b, p), stored = is_linear(v, v2, stored)
        if not p or a:
            stored[v1.name] = (False, False, False)
            return stored[v1.name], stored
    stored[v1.name] = (False, True, True)
    return stored[v1.name], stored

def linear(v1, v2, stored):
    """
        Linear coefficient and intercept extraction
        Return: (coefficient, intercept), stored
        stored: a dictionary to store intermediate results with the same v2. This trick is crucial for large models.
    """
    if v1.name in stored.keys():
        return stored[v1.name], stored
    if v1 == v2:
        return (Variable.all['1.'], Variable.all['0.']), stored
    if not v1.is_determinstic or v1.constant:
        return (Variable.all['0.'], v1), stored
    if v1.operator == 'add':
        (a, b), stored = linear(v1.operands[0], v2, stored)
        (c, d), stored = linear(v1.operands[1], v2, stored)
        if a is None or b is None or c is None or d is None:
            stored[v1.name] = (None, None)
            return stored[v1.name], stored
        if a == Variable.all['0.'] and c == Variable.all['0.']:
            x = Variable.all['0.']
        else:
            x = Variable()
            update([x], 'add', [a, c])
        if b == Variable.all['0.'] and d == Variable.all['0.']:
            y = Variable.all['0.']
        else:
            y = Variable()
            update([y], 'add', [b, d])
        stored[v1.name] = (x, y)
        return stored[v1.name], stored

    if v1.operator == 'sub':
        (a, b), stored = linear(v1.operands[0], v2, stored)
        (c, d), stored = linear(v1.operands[1], v2, stored)
        if a is None or b is None or c is None or d is None:
            stored[v1.name] = (None, None)
            return stored[v1.name], stored

        if a == Variable.all['0.'] and c == Variable.all['0.']:
            x = Variable.all['0.']
        else:
            x = Variable()
            update([x], 'sub', [a, c])
        if b == Variable.all['0.'] and d == Variable.all['0.']:
            y = Variable.all['0.']
        else:
            y = Variable()
            update([y], 'sub', [b, d])
        stored[v1.name] = (x,y)
        return stored[v1.name], stored

    if v1.operator == 'mul':
        (a, b), stored = linear(v1.operands[0], v2, stored)
        (c, d), stored = linear(v1.operands[1], v2, stored)
        if a is None or b is None or c is None or d is None:
            stored[v1.name] = (None, None)
            return stored[v1.name], stored
        if a == Variable.all['0.']:
            if b == Variable.all['0.'] or c == Variable.all['0.']:
                x = Variable.all['0.']
            else:
                x = Variable()
                update([x], 'mul', [b, c])
            if b == Variable.all['0.'] or d == Variable.all['0.']:
                y = Variable.all['0.']
            else:
                y = Variable()
                update([y], 'mul', [b, d])
            stored[v1.name] = (x, y)
            return stored[v1.name], stored
        if c == Variable.all['0.']:
            if a == Variable.all['0.'] or d == Variable.all['0.']:
                x = Variable.all['0.']
            else:
                x = Variable()
                update([x], 'mul', [a, d])
            if b == Variable.all['0.'] or d == Variable.all['0.']:
                y = Variable.all['0.']
            else:
                y = Variable()
                update([y], 'mul', [b, d])
            stored[v1.name] = (x, y)
            return stored[v1.name], stored
        stored[v1.name] = (None, None)
        return stored[v1.name], stored

    if v1.operator == 'div':
        (a, b), stored = linear(v1.operands[0], v2, stored)
        (c, d), stored = linear(v1.operands[1], v2, stored)
        if a is None or b is None or c is None or d is None:
            stored[v1.name] = (None, None)
            return stored[v1.name], stored
        if c == Variable.all['0.']:
            if a == Variable.all['0.']:
                x = Variable.all['0.']
            else:
                x = Variable()
                update([x], 'div', [a, d])
            if b == Variable.all['0.']:
                y = Variable.all['0.']
            else:
                y = Variable()
                update([y], 'div', [b, d])
            stored[v1.name] = (x, y)
            return stored[v1.name], stored
        stored[v1.name] = (None, None)
        return stored[v1.name], stored

    if v1.operator == 'register':
        return linear(v1.operands[0], v2, stored)

    for v in v1.operands:
        (a, b), stored = linear(v, v2, stored)
        if a is None or b is None or a != Variable.all['0.']:
            stored[v1.name] = (None, None)
            return stored[v1.name], stored
    stored[v1.name] = (Variable.all['0.'], v1)
    return stored[v1.name], stored


def is_dependent(v1, v2, stored):
    """
        Dependency detection
        Return: whether or not v1 is dependent on v2, stored
        stored: a dictionary to store intermediate results with the same v2. This trick is crucial for large models.
    """
    if v1.name in stored.keys():
        return stored[v1.name], stored
    if v1 == v2:
        return True, stored
    if not v1.is_determinstic or v1.constant:
        return False, stored
    for v in v1.operands:
        r, stored = is_dependent(v, v2, stored)
        if r:
            stored[v1.name] = True
            return True, stored
    stored[v1.name] = False
    return False, stored

if __name__ == '__main__':
    a = Variable('a')
    b = Variable('b')
    print(a.id, b.id)
    update(None, 'iota', None)