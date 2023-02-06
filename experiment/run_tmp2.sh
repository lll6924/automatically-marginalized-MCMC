for n in 100 200 300 400 500
do
  python -m run.main --model TMP --model_parameters "{'N':'$n'}" --just_compile --no_marginalization
done