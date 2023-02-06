for rng_key in 0 1 2 3 4
do
  python -m run.hmc --model ElectricCompanyReparameterized --rng_key $rng_key
      python -m run.hmc --model PulmonaryFibrosisReparameterized --rng_key $rng_key
done