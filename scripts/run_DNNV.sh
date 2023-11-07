#!/bin/bash

. $GDVB/scripts/init_conda.sh
conda activate dnnv

dnnv $@
