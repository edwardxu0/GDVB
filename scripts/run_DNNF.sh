#!/bin/bash

. $GDVB/scripts/init_conda.sh
conda activate dnnf

python -m dnnf $@
