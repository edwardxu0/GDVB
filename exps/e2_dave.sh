#!/usr/bin/env bash

#python -m gdvb configs/dave_2x2.toml train 10
#python -m gdvb configs/dave_2x2.toml gen_props 10
#python -m gdvb configs/dave_2x2.toml verify 10


#python -m gdvb configs/dave_2x2_enu.toml train 10
#python -m gdvb configs/dave_2x2_enu2.toml train 10
#python -m gdvb configs/dave_2x2_enu3.toml train 10
#python -m gdvb configs/dave_2x2_enu4.toml train 10

#python -m gdvb configs/dave_2x2_enu.toml gen_props 10
#python -m gdvb configs/dave_2x2_enu2.toml gen_props 10
#python -m gdvb configs/dave_2x2_enu3.toml gen_props 10
#python -m gdvb configs/dave_2x2_enu4.toml gen_props 10

#python -m gdvb configs/dave_2x2_enu.toml verify 10
python -m gdvb configs/dave_2x2_enu2.toml verify 10
python -m gdvb configs/dave_2x2_enu3.toml verify 10
python -m gdvb configs/dave_2x2_enu4.toml verify 10
