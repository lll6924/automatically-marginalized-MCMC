import jax.numpy as jnp
from algorithm import hmc
from core import preprocess, sweep_and_swap, get_log_prob
from time import time
from numpyro.diagnostics import effective_sample_size
import numpy as np
import importlib
from jax import random, jit, grad, make_jaxpr

def marginalized_hmc(model, model_parameters, warm_up_steps, sample_steps, result_file, rng_key, protected, algorithm, plot = False, just_compile = False, no_marginalization = True, parallel = False):
    time1 = time()

    # preprocess the model into our representation
    rvs, name_mapping, expr_mapping, observed, candidates, values, variables = preprocess(model, model_parameters)

    # optimize by reversing edges
    recovery_stack = []
    if not no_marginalization:
       rvs, candidates, variables, recovery_stack = sweep_and_swap(rvs, name_mapping, expr_mapping, observed, candidates,
                                                                    values, variables, protected)

    # transform the representation into log density function
    latent_dims, all_latent_dims, log_prob, postprocess, recover = get_log_prob(rvs, name_mapping, expr_mapping,
                                                                                observed, candidates, values, variables,
                                                                                recovery_stack)
    if just_compile:
        jpr = make_jaxpr(log_prob)(jnp.zeros(latent_dims))
        print('Size of the Jaxprs of log_prob', len(jpr.eqns))
        jpr2 = make_jaxpr(grad(log_prob))(jnp.zeros(latent_dims))
        print('Size of the Jaxprs of grad(log_prob)', len(jpr2.eqns))
        time2 = time()
        g = jit(grad(log_prob))
        print(g(jnp.zeros(latent_dims)))
        time3 = time()
        print('compilation time: ',time3-time2)
        if no_marginalization:
            result_file += '_no_marginalization'
        with open(result_file+'_compile','w') as f:
            print(len(jpr.eqns), len(jpr2.eqns), time3-time2, file=f)
    else:
        time2 = time()

        samples, samples_with_key = hmc(log_prob, jnp.zeros(latent_dims), all_latent_dims, postprocess, recover, result_file, algorithm = algorithm, warm_up_steps = warm_up_steps, sample_steps = sample_steps, rng_key=rng_key, parallel=parallel)

        time3 = time()

        ess = effective_sample_size(np.array(samples))

        if plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            plt.rcParams['font.size'] = '24'
            plt.rcParams['pdf.fonttype'] = 42
            # fig, axs = plt.subplots(2, 2, figsize=(12,10))
            fig, axs = plt.subplots(1, 1, figsize=(6, 5))
            if model in ['EightSchools']:
                for i in range(1):
                    data = []
                    for mu, tau in zip(samples_with_key['mu'][0], samples_with_key['tau'][0]):
                        data.append({'mu': float(mu), 'logtau': np.log(float(tau)), "source": "HMC"})
                    data = pd.DataFrame(data)
                    # data.to_csv('icml2023/figure/eight_schools-HMCM.csv')
                    sns.jointplot(x=data.mu, y=data.logtau, kind="hex", color="#4CB391", marginal_kws={'element':"step"})
                    # sns.kdeplot(x="mu", y='logtau', data=data.sample(100000), levels=5, color="#beaed4", linewidths=2)
                    sns.kdeplot(x="mu", y='logtau', data=data, levels=5, color="#beaed4", linewidths=2)
                    plt.xlim([-8, 15])
                    plt.ylim([-3, 3])
                    plt.xlabel('$\\mu$')
                    plt.ylabel('$\\log\\tau$')
                    plt.tight_layout()
                    plt.savefig('icml2023/figure/eight_schools-HMCM.pdf')


        with open(result_file,'w') as f:
            print(time2-time1, file=f)
            print(time3-time1, file=f)
            print(ess, file=f)
            print(np.mean(ess)/(time3-time1), file=f)
            print(np.min(ess) / (time3 - time1), file=f)


