import jax.random as random
from numpyro.infer import MCMC, NUTS, HMC
from numpyro.diagnostics import print_summary, effective_sample_size
import importlib
import click
from jax import device_get
import os
import numpy as np
from utils import PythonLiteralOption
from time import time
import pathlib
import arviz
import numpyro
import jax
numpyro.set_host_device_count(4)
print(jax.local_device_count())
"""
    TODO: use Arviz to evaluate multiple chains
"""

@click.command()
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference')
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
@click.option('--rng_key', default=0)
@click.option('--plot', is_flag=True)
@click.option('--model_parameters', cls=PythonLiteralOption, default='{}', help = 'The parameters as a dictionary that feed into model construction')
@click.option('--algorithm', default='NUTS', help = 'The MCMC algorithm of Numpyro to use. Choose from [\'NUTS\', \'HMC\']')
def main(model, warm_up_steps, sample_steps,rng_key,plot,model_parameters, algorithm):
    parameter_settings = '_'.join(model_parameters.values())

    result_path = 'result/{}/{}_{}_{}/{}'.format(model,parameter_settings, warm_up_steps, sample_steps,'HMC')
    result_file = result_path + '/' + str(rng_key)
    p = pathlib.Path(result_path)
    p.mkdir(parents=True, exist_ok=True)


    model_name = model
    module = importlib.import_module('model')
    model = getattr(module, model_name)(**model_parameters)
    start_time = time()
    post_rng_key, rng_key = random.split(random.PRNGKey(rng_key))
    if algorithm == 'NUTS':
        nuts_kernel = NUTS(model.model, adapt_mass_matrix=True)
    elif algorithm == 'HMC':
        nuts_kernel = HMC(model.model, adapt_mass_matrix=True)
    else:
        raise ValueError('Unknown algorithm: {}'.format(algorithm))

    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps, num_chains=4)
    mcmc.run(rng_key, *model.args(), **model.kwargs(), extra_fields=('potential_energy','z_grad'))
    sites = mcmc._states[mcmc._sample_field]
    #print(sites)
    end_time = time()
    data = arviz.from_numpyro(mcmc)
    print(arviz.ess(data))
    print_summary(sites)
    overall_time = end_time - start_time
    esss = []
    for par in sites.keys():
        if not par.endswith("_base"):
            ess = effective_sample_size(device_get(sites[par]))

            if type(ess) == np.float64:
                esss.append(ess)
            else:
                esss.extend(ess)



    with open(result_file,'w') as f:
        print(overall_time, file=f)
        print(esss, file=f)
        print(np.mean(esss)/overall_time, file=f)
        print(np.min(esss) / overall_time, file=f)


    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.rcParams['font.size'] = '24'
        plt.rcParams['pdf.fonttype'] = 42
        # fig, axs = plt.subplots(2, 2, figsize=(12,10))
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        if model_name in ['EightSchools']:
            ess = effective_sample_size(device_get(sites['tau']))
            with open(result_file, 'w') as f:
                print(overall_time, ess, ess / overall_time, file=f)
            for i in range(1):
                data = []
                for mu, tau in zip(sites['mu'][0], sites['tau'][0]):
                    data.append({'mu': float(mu), 'logtau': np.log(float(tau)), "source": "HMC"})
                data = pd.DataFrame(data)
                # data.to_csv('icml2023/figure/eight_schools-HMC.csv')
                sns.jointplot(x=data.mu, y=data.logtau, kind="hex", color="#4CB391", marginal_kws={'element':"step"})
                # sns.kdeplot(x="mu", y='logtau', data=data.sample(1000), levels=5, color="#beaed4", linewidths=2)
                sns.kdeplot(x="mu", y='logtau', data=data, levels=5, color="#beaed4", linewidths=2)
                plt.xlim([-8, 15])
                plt.ylim([-3, 3])
                plt.xlabel('$\\mu$')
                plt.ylabel('$\\log\\tau$')
                plt.tight_layout()
                plt.savefig('icml2023/figure/eight_schools-HMC.pdf')
        if model_name == 'GammaModel':
            data = []
            for mu, tau in zip(sites['x1'][0], sites['x2'][0]):
                data.append({'x1': float(mu), 'x2': np.log(float(tau)), "source": "HMC"})
            data = pd.DataFrame(data)
            sns.scatterplot(x="x1", y='x2', data=data, s=5, color=".15")
            sns.histplot(x="x1", y='x2', data=data, bins=50, pthresh=.1, cmap="mako")
            sns.kdeplot(x="x1", y='x2', data=data, levels=5, color="w", linewidths=1)
            plt.xlabel('$x1')
            plt.ylabel('$x2$')
            plt.tight_layout()
            plt.show()
            #plt.savefig('writing/figure/out.png')



if __name__ == "__main__":
    main()
