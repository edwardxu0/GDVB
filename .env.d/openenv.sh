#!/bin/bash

source .venv/bin/activate

# GDVB
export GDVB=`pwd`
export TMPDIR=$GDVB/tmp

# r4v
export PYTHONPATH="${PYTHONPATH}:${GDVB}"
export R4V="${GDVB}/lib/R4V"
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_wb="${GDVB}/lib/DNNV_wb"
export DNNF="${GDVB}/lib/DNNF"
