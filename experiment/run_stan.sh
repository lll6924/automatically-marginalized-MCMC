for rng_key in 3 4 5
do
  #python -m run.stan_hpp --dataset rat_tumors --seed $rng_key
  #python -m run.stan_hpp --dataset baseball_large --seed $rng_key
  #python -m run.stan_hpp --dataset baseball_small --seed $rng_key
  python -m run.stan_hpp_marginalized --dataset rat_tumors --seed $rng_key
  python -m run.stan_hpp_marginalized --dataset baseball_large --seed $rng_key
  # python -m run.stan_hpp_marginalized --dataset baseball_small --seed $rng_key
done