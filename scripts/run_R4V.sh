#!/bin/bash


. $GDVB/scripts/init_conda.sh
. $R4V/.env.d/openenv.sh

python -m r4v $@

