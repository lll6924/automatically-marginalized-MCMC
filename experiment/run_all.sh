for rng_key in 0 1 2 3 4
do
  python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'rat_tumors'}" --rng_key $rng_key
  python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_large'}" --rng_key $rng_key
  python -m run.main --model HierarchicalPartialPooling --model_parameters "{'dataset':'baseball_small'}" --rng_key $rng_key
  python -m run.main --model ElectricCompany --rng_key $rng_key --protected "['mua0','mua1','mua2','mua3']"
  python -m run.main --model PulmonaryFibrosis --rng_key $rng_key --protected "['m_a','s_a','m_b','s_b']"
done