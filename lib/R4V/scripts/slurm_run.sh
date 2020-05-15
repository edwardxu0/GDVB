#!/usr/bin/env bash
#SBATCH --job-name=distillation

# print out some debug info
date
echo $(hostname)
echo "[PATH]/slurm_run.sh $@"

# set up environment
. .env.d/openenv.sh

# prepare the config file
filename=$(basename $1)
identifier=$2
echo "$identifier"

filename=$(basename $1)
config_name="${filename%.*}"
echo $config_name
mkdir -p tmp/$config_name/$identifier
config=tmp/$config_name/$identifier/config.toml
echo $config
cat $1 > $config
echo "[distillation.student]" >> $config
echo "path=\"tmp/$config_name/$identifier/model.onnx\"" >> $config

echo

# run distillation
echo "python -m r4v distill $config -v"
python -m r4v distill $config -v
