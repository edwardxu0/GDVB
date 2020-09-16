#!/bin/bash

source .venv/bin/activate

# GDVB
export GDVB=`pwd`
export TMPDIR=$GDVB/tmp

# r4v
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"
export DNNV="${GDVB}/lib/DNNV"
