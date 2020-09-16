#!/bin/bash

source .venv/bin/activate

# GDVB
export GDVB=`pwd`

# r4v
export PYTHONPATH="${PYTHONPATH}:${GDVB}/lib/R4V/"
export DNNV="${GDVB}/lib/DNNV"
