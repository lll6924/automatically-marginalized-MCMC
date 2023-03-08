import importlib

import jax.core
from jax import make_jaxpr, random
from jax.core import CallPrimitive
from numpyro.handlers import seed, trace

from utils import get_alphabetic_list
from rule import Variable, update
import jax.numpy as jnp
from primitives import MyUniform, MyPareto, distribution_mapping, register_p

def preprocess(model, model_parameters):
    module = importlib.import_module('model')
    model = getattr(module, model)(**model_parameters)
    key = random.PRNGKey(0)
    model_seeded = seed(model.model, key)
    model_trace = trace(model_seeded).get_trace(*model.args(), **model.kwargs())

    # get variable names and ordering in NumPyro
    rvs = []
    values = {}
    observed = {}
    candidates = set()
    for site in model_trace.values():
        if site["type"] == 'sample':
            values[site['name']] = site['value']
            candidates.add(site['name'])
            if site['is_observed']:
                observed[site['name']] = True
            else:
                observed[site['name']] = False
            dist = site['fn'].__class__.__name__
            if dist == 'ExpandedDistribution':
                dist = site['fn'].base_dist.__class__.__name__
            rvs.append({'name': site['name'], 'dist': dist})
    jpr = make_jaxpr(model_seeded)(*model.args())# , **model.kwargs())
    #with open('tmp','w') as f:
    #    print(jpr,file = f)
    eqns = jpr.eqns
    # print(jpr.jaxpr)
    # loop over Jaxprs, identify the random variables
    is_rv = {}
    dep = {}
    pred = {}
    car = {}
    n_consts = len(jpr.consts)
    n_in_vars = len(jpr.in_avals)
    for par in get_alphabetic_list(n_consts + n_in_vars):
        dep[par] = set()
        car[par] = set()
        is_rv[par] = False
    rv_id = 0
    expr_mapping = {}
    name_mapping = {}
    for e in eqns:
        if str(e.primitive) == 'register':
            for var in e.invars:
                var = str(var)
                is_rv[var] = True
                car[var] = {var}
                rvs[rv_id]['expr'] = var
                expr_mapping[var] = rv_id
                name_mapping[rvs[rv_id]['name']] = rv_id
                rv_id += 1

        for var in e.outvars:
            is_rv[str(var)] = False
            var = str(var)
            dep[var] = set()
            pred[var] = []
            for prev in e.invars:
                if isinstance(prev,jax.core.Literal):
                    prev = str(prev.val) + str(prev.aval)
                else:
                    prev = str(prev)
                if prev in car.keys():
                    dep[var] = dep[var].union(car[prev])
                pred[var].append(prev)
            car[var] = dep[var]

    for k, w in dep.items():
        if is_rv[k]:
            rvs[expr_mapping[k]]['dep'] = []
            for d in w:
                if is_rv[d]:
                    rvs[expr_mapping[k]]['dep'].append(rvs[expr_mapping[d]]['name'])
    variables = {}
    variables['1.'] = Variable('1.', 1.)
    variables['1'] = Variable('1', jnp.array(1, dtype=jnp.int32))
    variables['0.'] = Variable('0.', 0.)
    variables['_'] = Variable('_')
    # print(n_consts, n_in_vars, key)
    alphabetic_list = get_alphabetic_list(n_consts + n_in_vars)
    for i in range(n_consts):
        variables[alphabetic_list[i]] = Variable(alphabetic_list[i], jpr.consts[i])

    lists = list(model.args())
    lists.extend(model.kwargs().values())
    for i in range(n_in_vars):
        variables[alphabetic_list[i + n_consts]] = Variable(alphabetic_list[i + n_consts], lists[i])
    for e in eqns:
        #print(e.primitive,e.invars, e.outvars)
        ins = []
        for v in e.invars:
            if not str(v) in variables.keys():
                variables[str(v.val)+str(v.aval)] = Variable(str(v.val)+str(v.aval), v.val)
                ins.append(variables[str(v.val)+str(v.aval)])
            else:
                ins.append(variables[str(v)])
        for var in e.outvars:
            var = str(var)
            if var == '_':
                continue
            deps = {variables[d] for d in dep[var]}
            v = Variable(var, None, not is_rv[var], ins, deps)
            variables[var] = v

        outs = [variables[str(v)] for v in e.outvars]
        if str(e.primitive) == 'register':
            update(outs, str(e.primitive), ins, params=e.params)
        else:
            update(outs, str(e.primitive), ins, eqn=e)

    # print(variables)
    # for e in eqns:
    #     outs = [variables[str(v)] for v in e.outvars]
    #     ins = [variables[str(v)] for v in e.invars]
    #     if str(e.primitive) == 'xla_call' and e.params['name'] == 'register':
    #         update(outs, str(e.primitive), ins, params=e.params)
    #     else:
    #         update(outs, str(e.primitive), ins, eqn=e)
        # rint(outs[0].value)
    # values = {'mu': jnp.asarray(0.), 'tau': jnp.asarray(3.), 'theta': jnp.zeros(8), 'obs': model.kwargs()['obs']}
    # for e in eqns:
    #      print(e.primitive, e.invars, e.outvars, e.params)
    for rv in rvs:
        assert(rv['dist'] in distribution_mapping.keys())
        if rv['dist'] == 'Uniform':
            alpha, beta = MyUniform.get_parameters(pred, rv)
            assert (alpha == '0.0ShapedArray(float32[])' and beta == '1.0ShapedArray(float32[])')
        if rv['dist'] == 'Pareto':
            alpha, beta = MyPareto.get_parameters(pred, rv)
            assert (alpha == '1.0ShapedArray(float32[])')
        parameter_names = distribution_mapping[rv['dist']].get_parameters(pred, rv)
        tup = tuple()
        for p in parameter_names:
            tup = tup + (variables[p],)
        variables[rv['expr']].parameters = tup

    candidates = set()
    for rv in rvs:
        rv['children'] = []
        candidates.add(rv['name'])
    for rv in rvs:
        for d in rv['dep']:
            rvs[name_mapping[d]]['children'].append(rv['name'])

    return rvs, name_mapping, expr_mapping, observed, candidates, values, variables