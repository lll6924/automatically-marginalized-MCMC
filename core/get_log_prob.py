import jax.numpy as jnp
import jax.scipy as jsc
from primitives import MyNormal, MyHalfCauchy, MyGamma, MyExponential, MyCompoundGamma, MyBeta, MyBernoulli, MyBinomial, MyBetaBinomial, MyUniform, MyPareto, distribution_mapping
from jax import random, jit, grad, make_jaxpr
from time import time

def get_log_prob(rvs, name_mapping, expr_mapping, observed, candidates, values, variables, recovery_stack):
    to_evaluate = {}
    for key, value in values.items():
        to_evaluate[rvs[name_mapping[key]]['expr']] = value
    candidates = sorted(candidates, key=lambda x: observed[x])
    all_obs = []
    for c in candidates:
        if observed[c]:
            all_obs.append(values[c].flatten())
    all_obs = jnp.concatenate(all_obs)
    def evaluate2(substitute):
        res = 0.
        stored = {}
        for c in candidates:
            r = rvs[name_mapping[c]]
            if r['name'] in candidates:
                params, stored = variables[r['expr']].get_parameters(substitute=substitute, stored = stored)
                res += jnp.sum(distribution_mapping[r['dist']].evaluate(params, substitute[r['expr']]))

        return res

    dims = {}
    last = 0
    latent_dims = 0
    for key in candidates:
        value = values[key]
        value_shape = value.shape
        all = jnp.prod(jnp.array(value_shape), dtype=int)
        if not observed[key]:
            latent_dims += all
        dims[key] = {'left': last, 'right': last + all, 'shape': value_shape}
        last = last + all

    recovered_last = latent_dims
    for key in reversed(recovery_stack):
        value = values[key]
        value_shape = value.shape
        all = jnp.prod(jnp.array(value_shape), dtype=int)
        dims[key] = {'left': recovered_last, 'right': recovered_last + all, 'shape': value_shape}
        recovered_last = recovered_last + all

    def translate(z):
        to_evaluate = {}
        cumulated = 0.
        for key in candidates:
            val = z[dims[key]['left']:dims[key]['right']]
            if rvs[name_mapping[key]]['dist'] in ['HalfCauchy', 'Gamma', 'Exponential', 'CompoundGamma'] and not \
            observed[key]:
                cumulated += jnp.sum(val)
                val = jnp.exp(val)
            elif rvs[name_mapping[key]]['dist'] in ['Beta', 'Uniform'] and not observed[key]:
                val = jsc.special.expit(val)
                cumulated += jnp.sum(jnp.log(val) + jnp.log(1 - val))
            elif rvs[name_mapping[key]]['dist'] in ['Pareto'] and not observed[key]:
                cumulated += jnp.sum(val)
                val = jnp.exp(val) + 1.
            # print(val.shape,key, dims[key]['shape'])
            to_evaluate[rvs[name_mapping[key]]['expr']] = jnp.reshape(val, dims[key]['shape'])
        return to_evaluate, cumulated

    def log_prob(x):
        z = jnp.concatenate([x, all_obs])
        to_evaluate, cumulated = translate(z)
        return evaluate2(substitute=to_evaluate) + cumulated
    # jit1 = time()
    # jt = jit(log_prob)
    # print(jt(jnp.zeros(latent_dims)))
    # jtt = jit(grad(jt))
    # print(jtt(jnp.zeros(latent_dims)))
    # jit2 = time()
    # print(jit2-jit1)
    def postprocess(sites):
        res = {}
        for key in candidates:
            if not observed[key]:
                # print(key, dims[key]['left'], dims[key]['right'])
                val = sites[:, :, dims[key]['left']:dims[key]['right']]

                if rvs[name_mapping[key]]['dist'] in ['HalfCauchy', 'Gamma', 'Exponential', 'CompoundGamma']:
                    val = jnp.exp(val)
                elif rvs[name_mapping[key]]['dist'] in ['Beta', 'Uniform']:
                    val = jsc.special.expit(val)
                elif rvs[name_mapping[key]]['dist'] in ['Pareto']:
                    val = jnp.exp(val) + 1.
                res[key] = val
        for key in reversed(recovery_stack):
            val = sites[:, :, dims[key]['left']:dims[key]['right']]
            res[key] = val
        return res
    def recover(x, rng_key):
        z = jnp.concatenate([x, all_obs])
        substitute, _ = translate(z)
        recovered = []
        stored = {}
        for v in reversed(recovery_stack):
            r = rvs[name_mapping[v]]
            rng_key, this_key = random.split(rng_key)

            params, stored = variables[r['expr']].get_parameters(substitute=substitute, stored=stored)
            val = distribution_mapping[r['dist']].sample(params, this_key)

            substitute[r['expr']] = jnp.reshape(val, values[v].shape)
            recovered.append(val.flatten())
        if len(recovered)>0:
            recovered = jnp.concatenate(recovered)
            return jnp.concatenate([x, recovered])
        else:
            return x

    #print(log_prob(jnp.zeros(latent_dims)))

    return latent_dims, recovered_last, log_prob, postprocess, recover