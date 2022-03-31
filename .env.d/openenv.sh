#!/bin/bash

source .venv/bin/activate

# GDVB
if [ -z ${GDVB} ]; then
  export GDVB=`pwd`
fi
export TMPDIR=$GDVB/tmp

# libraries
export R4V="${GDVB}/lib/R4V"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"

# path
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"

# misc
export acts_path="${GDVB}/lib/acts.jar"
export GRB_LICENSE_FILE="${GDVB}/lib/gurobi.lic"

alias gdvb="python -m gdvb"