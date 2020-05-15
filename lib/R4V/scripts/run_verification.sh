#!/bin/bash

export GRB_LICENSE_FILE=/u/dls2fc/gurobi_lic/$(hostname)/gurobi.lic
export DNNA=/p/d4v/dls2fc/dnna
export PYTHONPATH=$PYTHONPATH:$DNNA
export PATH=$PATH:$DNNA/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DNNA/lib

results_csv=$1; shift
model_dir=$1; shift
property_csv=$1; shift
verifier=$1; shift

echo "python -u ./tools/run_verification.py $results_csv $model_dir $property_csv $verifier $@"
python -u ./tools/run_verification.py $results_csv $model_dir $property_csv $verifier $@
