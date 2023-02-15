import click
import os
from core import marginalized_hmc
from utils import PythonLiteralOption
import sys
import pathlib
import numpyro
import jax
numpyro.set_host_device_count(4)
sys.setrecursionlimit(10000)
print(jax.local_device_count())
@click.command()
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference. See classes under model/ for details')
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
@click.option('--rng_key', default=0)
@click.option('--protected', cls=PythonLiteralOption, default='[]', help = 'The variables that should not be marginalized')
@click.option('--model_parameters', cls=PythonLiteralOption, default='{}', help = 'The parameters as a dictionary that feed into model construction')
@click.option('--algorithm', default='NUTS', help = 'The MCMC algorithm of Numpyro to use. Choose from [\'NUTS\', \'HMC\']')
@click.option('--plot', is_flag=True)
@click.option('--just_compile', is_flag=True, help = 'If flagged, the number of Jaxprs will be reported')
@click.option('--no_marginalization', is_flag=True, help = 'If flagged, automatic marginalization will not be performed')
def main(model, warm_up_steps, sample_steps, rng_key, protected, model_parameters, algorithm, plot, just_compile,no_marginalization):
    print('Protected variables:', protected)
    if not os.path.exists('result'):
        os.mkdir('result')
    parameter_settings = '_'.join(model_parameters.values())
    parameter_settings = parameter_settings + '_{}'.format('-'.join(protected))
    result_path = 'result/{}/{}_{}_{}/{}'.format(model,parameter_settings, warm_up_steps, sample_steps,'MHMC')
    result_file = result_path + '/' + str(rng_key)
    p = pathlib.Path(result_path)
    p.mkdir(parents=True, exist_ok=True)

    marginalized_hmc(model, model_parameters, warm_up_steps, sample_steps, result_file, rng_key, protected, algorithm, plot, just_compile,no_marginalization)






if __name__ == '__main__':
    main()