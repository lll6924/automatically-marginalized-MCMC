import click
import os
from core import marginalized_hmc
from utils import PythonLiteralOption
import sys
sys.setrecursionlimit(10000)

@click.command()
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference.')
@click.option('--warm_up_steps', default=10000)
@click.option('--sample_steps', default=100000)
@click.option('--rng_key', default=0)
@click.option('--protected', cls=PythonLiteralOption, default='[]', help = 'The variables that should not be marginalized')
@click.option('--model_parameters', cls=PythonLiteralOption, default='{}', help = 'The parameters that feed into model construction')
@click.option('--algorithm', default='NUTS', help = 'The MCMC algorithm of Numpyro to use')
@click.option('--plot', is_flag=True)
@click.option('--just_compile', is_flag=True)
@click.option('--no_marginalization', is_flag=True)
def main(model, warm_up_steps, sample_steps, rng_key, protected, model_parameters, algorithm, plot, just_compile,no_marginalization):
    print('Protected variables:', protected)
    if not os.path.exists('result'):
        os.mkdir('result')
    parameter_settings = '_'.join(model_parameters.values())
    parameter_settings = parameter_settings + '_{}'.format('-'.join(protected))
    if not os.path.exists('result/{}'.format(model)):
        os.mkdir('result/{}'.format(model))
    if not os.path.exists('result/{}/{}_{}_{}'.format(model,parameter_settings, warm_up_steps, sample_steps)):
        os.mkdir('result/{}/{}_{}_{}'.format(model,parameter_settings, warm_up_steps, sample_steps))
    if not os.path.exists('result/{}/{}_{}_{}/{}'.format(model,parameter_settings, warm_up_steps, sample_steps,'MHMC')):
        os.mkdir('result/{}/{}_{}_{}/{}'.format(model,parameter_settings, warm_up_steps, sample_steps,'MHMC'))
    result_file = 'result/{}/{}_{}_{}/{}/{}'.format(model,parameter_settings, warm_up_steps, sample_steps,'MHMC',str(rng_key))

    marginalized_hmc(model, model_parameters, warm_up_steps, sample_steps, result_file, rng_key, protected, algorithm, plot, just_compile,no_marginalization)






if __name__ == '__main__':
    main()