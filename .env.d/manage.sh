#!/bin/bash

if [ "$1" == "init" ]
then
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    while read req || [ -n "$req" ]
    do
        echo "pip install $req"
        pip install $req
    done < .env.d/requirements.txt
fi

if [ "$1" == "install" ]
then
    shift
    for pkg in "$@"
	do
		if [ "$pkg" == "R4V" ]
        then
			echo "R4V is included ..."
			echo "Done."
        elif [ "$pkg" == "DNNV" ]
        then
			echo "Downloading DNNV..."
			cd lib
			git clone https://github.com/dlshriver/DNNV.git
			cd DNNV
			./manage.sh install reluplex planet mipverify neurify eran
			cd ../../
			echo "Done."
        else
            echo "Unknown package: $pkg"
        fi
    done
fi
