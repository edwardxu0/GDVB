#!/usr/bin/env bash

#python -m gdvb configs/dave_2x2.toml train 10
#python -m gdvb configs/dave_2x2.toml train 11
#python -m gdvb configs/dave_2x2.toml train 12
#python -m gdvb configs/dave_2x2.toml train 13
#python -m gdvb configs/dave_2x2.toml train 14

#python -m gdvb configs/dave_2x2.toml gen_props 10
#python -m gdvb configs/dave_2x2.toml gen_props 11
#python -m gdvb configs/dave_2x2.toml gen_props 12
#python -m gdvb configs/dave_2x2.toml gen_props 13
#python -m gdvb configs/dave_2x2.toml gen_props 14

python -m gdvb configs/dave_2x2.toml verify 10
python -m gdvb configs/dave_2x2.toml verify 11
python -m gdvb configs/dave_2x2.toml verify 12
python -m gdvb configs/dave_2x2.toml verify 13
python -m gdvb configs/dave_2x2.toml verify 14
