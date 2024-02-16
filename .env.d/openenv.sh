#!/bin/bash

conda activate gdvb

# GDVB
if [ -z ${GDVB} ]; then
  export GDVB=`pwd`
fi

export ROOT=`pwd`

# libraries
export R4V="${GDVB}/lib/R4V"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"
export SwarmHost="${GDVB}/lib/SwarmHost"

# path
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export PYTHONPATH="${PYTHONPATH}:${R4V}"
export PYTHONPATH="${PYTHONPATH}:${SwarmHost}"

# misc
export acts_path="${GDVB}/lib/acts.jar"
export GRB_LICENSE_FILE="${GDVB}/lib/gurobi.lic"

alias gdvb="python -m gdvb"
