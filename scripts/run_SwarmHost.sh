#!/bin/bash

. $GDVB/scripts/init_conda.sh
. $SwarmHost/.env.d/openenv.sh

python -m swarm_host $@

