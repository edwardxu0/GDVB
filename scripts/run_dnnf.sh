#!/bin/bash

cd $DNNF
. $DNNF/.env.d/openenv.sh
cd $GDVB

python -m dnnf $@
