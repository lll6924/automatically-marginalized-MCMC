from pystan import stan
import pystan
import multiprocessing

multiprocessing.set_start_method("fork")
from time import time
import click
import os
import pathlib
import numpy as np
from dataset.repeated_binary_trials import get_repeated_binary_trials_data
model_code = """
    data {
        int<lower=0> N;   
        int<lower=0> K[N];        
        int<lower=0> y[N];                   
    }
    parameters {
        real<lower=0, upper=1> phi;         // population chance of success
        real<lower=1> kappa;                // population concentration
        vector<lower=0, upper=1>[N] theta;    
    }
    
    model {
        kappa ~ pareto(1, 1.5);                        // hyperprior
        theta ~ beta(phi * kappa, (1 - phi) * kappa);  // prior
        y ~ binomial(K, theta);                        // likelihood
    }
"""

def getess(fit):
    return np.min(np.array(pystan.misc._summary(fit)['summary'])[:,-2])


@click.command()
@click.option('--dataset', default='baseball_small')
@click.option('--seed', default=0)
def main(dataset, seed):
    x, y, test_x, test_y = get_repeated_binary_trials_data(dataset)
    N = len(x)
    data = {'N':N, 'K':x, 'y':y}
    time1 = time()

    sm = pystan.StanModel(model_code = model_code)
    fit = sm.sampling(data=data, chains=1, warmup=10000, iter=110000, seed = seed)
    eta = fit.extract()["kappa"]
    time2 = time()
    print(pystan.stansummary(fit,pars='kappa'))

    ess_min = getess(fit)

    print(ess_min)
    print(time2 - time1)

    result_path = 'result/StanHPP/{}/'.format(dataset)
    result_file = result_path + '/' + str(seed)
    p = pathlib.Path(result_path)
    p.mkdir(parents=True, exist_ok=True)
    with open(result_file,'w') as f:
        print(ess_min, time2-time1,ess_min/(time2-time1),file=f)
    #from matplotlib import pyplot as plt
    #import seaborn as sns

    #sns.displot(eta)
    #plt.show()


if __name__ == '__main__':
    main()