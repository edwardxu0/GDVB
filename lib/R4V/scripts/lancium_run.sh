#!/usr/bin/env bash

# print out some debug info
date
echo $(hostname)
echo "$0 $@"

# move to app directory
ln -s $(pwd)/scratch/r4v/artifacts r4v/artifacts
. scratch/.venv/bin/activate
echo $(python -V)
echo $(which python)
cd r4v

# prepare the config file
echo "Preparing configuration file..."
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
echo "Running distillation..."
echo "python -m r4v distill $config -v"
python -m r4v distill $config -v

tar -czf ../$identifier.model.tar.gz tmp