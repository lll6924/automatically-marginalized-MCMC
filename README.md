# Automatic_Marginalization

## Dependencies

* JAX 0.4.1
* NumPyro 0.10.1

## Running Commands

To reproduce the results with HMC-M, the following commands could be used

```bash
python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'rat_tumors'}" --rng_key $rng_key
python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_large'}" --rng_key $rng_key
python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_small'}" --rng_key $rng_key
python -m run.main --model ElectricCompany --rng_key $rng_key --protected "['mua0','mua1','mua2','mua3']"
python -m run.main --model PulmonaryFibrosis --rng_key $rng_key --protected "['m_a','s_a','m_b','s_b']"
```

To reproduce the results with HMC, the following commands could be used

```bash
python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'rat_tumors'}" --rng_key $rng_key
python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_large'}" --rng_key $rng_key
python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_small'}" --rng_key $rng_key
python -m run.hmc --model ElectricCompany --rng_key $rng_key
python -m run.hmc --model PulmonaryFibrosisVectorized --rng_key $rng_key
```

To reproduced the results with HMC-R, please use

```bash
python -m run.hmc --model ElectricCompanyReparameterized --rng_key $rng_key
python -m run.hmc --model PulmonaryFibrosisReparameterized --rng_key $rng_key
```

To reproduce the results in Appendix F, run the following commands with `$n` set to the number of branches. 

```bash
python -m run.main --model TMP --model_parameters "{'N':'$n'}" --just_compile
python -m run.main --model TMP --model_parameters "{'N':'$n'}" --just_compile --no_marginalization
```

All above commands could be found under `experiment/`.

## Additional Notes

Our codes depend on the patterns of Jaxprs when tracing a program, which could 
be different for different versions of JAX/NumPyro and different system environments,
but should work in most cases.