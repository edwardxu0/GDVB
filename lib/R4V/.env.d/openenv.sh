#!/bin/bash

if [ -z ${DISPLAY:x} ]
then
    . ~/.bashrc
fi

if [ -e ./.venv/bin/activate ]
then
    . .venv/bin/activate
fi

export ENV_OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export ENV_OLD_PATH=$PATH
export ENV_OLD_PYTHONPATH=$PYTHONPATH

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)

export PYTHONPATH=$PROJECT_DIR/lib:$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

export PATH=$PROJECT_DIR/bin:$PATH
