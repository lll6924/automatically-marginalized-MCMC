import jax.random as random
from numpyro.infer import MCMC, NUTS, HMC
import jax.numpy as jnp
from numpyro.diagnostics import print_summary, effective_sample_size
from jax import device_get
from jax.lax import fori_loop
import arviz
import numpyro
def hmc(log_prob, init_params, all_latent_dims, postprocess = None, recover = None,
        result_file = 'result', plot = False, warm_up_steps = 10000, sample_steps = 100000, rng_key = 0,
        algorithm = 'NUTS', parallel=False):

    """
        Call NumPyro's HMC given a log density function, and recover with a recovery function
        TODO: use Arviz to evaluate multiple chains
    """

    def potential(x):
        return -log_prob(x)
    post_rng_key, rng_key = random.split(random.PRNGKey(rng_key))
    if algorithm == 'NUTS':
        nuts_kernel = NUTS(potential_fn=potential, adapt_mass_matrix=True, )
    elif algorithm == 'HMC':
        nuts_kernel = HMC(potential_fn=potential, adapt_mass_matrix=True, )
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))
    num_chains = 1
    if parallel:
        num_chains = 4
        init_params = jnp.repeat(jnp.expand_dims(init_params, axis=0), num_chains ,axis=0)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps, num_chains=num_chains)
    mcmc.run(rng_key, extra_fields=('potential_energy','z_grad'),init_params = init_params)
    sites = mcmc._states[mcmc._sample_field]
    if recover is not None:
        new_sites = []
        for j in range(len(sites)):
            this_site = sites[j]
            updated_sites = jnp.zeros((sample_steps,all_latent_dims))
            init_val = (post_rng_key, updated_sites)
            def fun(i, val):
                rng_key, updated_sites = val
                rng_key, cur_key = random.split(rng_key)
                res = recover(this_site[i],cur_key)
                updated_sites = updated_sites.at[i].set(res)
                return (rng_key, updated_sites)
            _, this_site = fori_loop(0, sample_steps, fun, init_val)
            new_sites.append(this_site)
        sites = jnp.array(new_sites)
    if postprocess is not None:
        sites_kv = postprocess(sites)
        data = arviz.from_dict(sites_kv)
        print(arviz.ess(data))
        print_summary(sites_kv)
        return sites, sites_kv
    else:
        print_summary(sites)
        return sites, None




if __name__ == '__main__':
    def log_prob(x):
        return jnp.sum(-jnp.square(x))
    hmc(log_prob, jnp.zeros(2))