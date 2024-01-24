#!/bin/bash

#. $DNNV/.venv/bin/activate
. $GDVB/scripts/init_conda.sh
conda activate dnnv

dnnv $@
