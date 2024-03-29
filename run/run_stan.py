from pystan import stan
import pystan
import multiprocessing

multiprocessing.set_start_method("fork")
from time import time
import click
import numpy as np
import numpyro.distributions as dist
import jax.numpy as jnp


@click.command()
@click.option('--n', default=100)
def main(n):
    #schools_data = {"a": [28.,  8., -3.,  7., -1.,  1., 18., 12.], "b": [15., 10., 16., 11.,  9., 11., 10., 18.]}
    schools_data = {"a0": [ 84.9,  97.2, 104.8,  77. , 113.3,  84.2, 107.2, 114.1,
             108.3, 116.9, 105.8,  98.9, 115.9, 105.8,  98.9,  95.1,
             113.9,  80.4, 101.2, 111.7,  72.7,  59.3,  87.9, 110.9,
             113.4, 111.5,  76.6, 111. , 114.2, 109.7,  97.3, 116.2,
              92.4,  71.1,  88.5,  70.5,  86.8, 102.5, 116.2,  93.4,
              97.3, 115.7,  94.9, 115.2,  75.6,  66.4, 110. ,  86. ,
              80. , 102.9,  72.9,  94.4, 103. ,  44.2, 115.3,  92.8,
             103.1, 101.3,  78.9, 113.1,  92.5,  69.1, 101.3, 114. ,
             101.9,  74.1,  60.8,  81.6, 111.7,  77.5, 109.4,  94. ,
             112.1, 110.6,  83.6,  99.6,  98. , 101.9, 106.9, 111.1,
             115.3, 111. ,  94.4,  97.8,  98.9, 115.6,  98.3, 102.2,
             101.9, 101. ,  54.6,  55.5, 110.6, 119.6, 110.8, 106.4,
              96.2, 116.6,  78.4,  96.5,  95.5, 107.6, 104.9, 111.2,
             105.6, 114.8, 109.6,  84.5,  60.6, 101.2,  93.6,  75.7,
             112.4, 100.6,  74.1, 103.7,  96.9,  55.3, 110.6, 104.6,
              89.5, 110.6, 100.8, 105.4,  95.3, 118. , 116.2,  76.3,
             102.9,  84.8, 122. ,  68.9,  67.6, 114.3,  52.3, 114.7,
             109.6, 105.8, 104.5, 103.6,  84.7,  90.5, 111.3, 110.4,
             114. , 100.1, 112.2,  87. , 108.6, 110.3, 111.6, 103.8,
              75.2,  69.7,  82.4, 115.5, 115.9, 110.2,  89.7, 109.2,
             107.1, 119.7, 111.2,  97.2, 114.4, 108.9, 113.8, 114.5,
             114.6,  88.9,  96. ,  73.7, 110.3,  52.9,  55. , 109.9,
             102.4,  85.3,  48.9, 114.9, 104.9, 108.9, 113.6,  91.7,
              47. ,  91.7, 113.9,  56.5,  70.6, 103.9, 113.3, 104.8]}
    print(len(schools_data['a0']))
    time1 = time()

    sm = pystan.StanModel(file = 'compiled')
    time2 = time()
    print(time2 - time1)
    fit = sm.sampling(data=schools_data, chains=1, iter=100000)
    eta = fit.extract()["c"]
    print(pystan.stansummary(fit))
    from matplotlib import pyplot as plt
    import seaborn as sns

    sns.displot(eta)
    plt.show()


if __name__ == '__main__':
    main()