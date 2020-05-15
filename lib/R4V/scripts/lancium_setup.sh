#!/usr/bin/env bash

# print out some debug info
date
echo $(hostname)
echo "$0 $@"

# set up environment (do once)
if [ ! -e ./scratch/.venv/bin/activate ]
then
    cd scratch
    echo "Setting up execution environment..."
    echo $(python -V)
    echo $(which python)
    python -m venv .venv
    . .venv/bin/activate
    echo $(python -V)
    echo $(which python)

    while read req || [ -n "$req" ]
    do
        echo "pip install $req"
        pip install $req
    done < ../r4v/requirements.txt
    deactivate
    cd ..
fi