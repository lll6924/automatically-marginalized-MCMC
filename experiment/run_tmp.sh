for n in 50 100 150 200 250
do
  python -m run.main --model TMP --model_parameters "{'N':'$n'}" --just_compile
done