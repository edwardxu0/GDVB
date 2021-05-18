#!/bin/bash

source .venv/bin/activate

# GDVB
export GDVB=`pwd`
export TMPDIR=$GDVB/tmp

# r4v
export R4V="${GDVB}/lib/R4V"
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"
export DNNV="${GDVB}/lib/DNNV"
export DNNV_ins="${GDVB}/lib/DNNV_ins"
export DNNF="${GDVB}/lib/DNNF"
