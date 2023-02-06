for rng_key in 0 1 2 3 4
do
  python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'rat_tumors'}" --rng_key $rng_key
  python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_large'}" --rng_key $rng_key
  python -m run.hmc --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_small'}" --rng_key $rng_key
  python -m run.hmc --model ElectricCompany --rng_key $rng_key
  python -m run.hmc --model PulmonaryFibrosisVectorized --rng_key $rng_key
done