#!/bin/bash

for i in {10..14}
do
	python -m gdvb configs/mcb.toml gen_ca $i
	python -m gdvb configs/mcb.toml train $i
	python -m gdvb configs/mcb.toml gen_props $i
	python -m gdvb configs/mcb.toml verify $i
	python -m gdvb configs/mcb.toml analyze $i
done

for i in {10..14}
do
	python -m gdvb configs/dave.toml gen_ca $i
	python -m gdvb configs/dave.toml train $i
	python -m gdvb configs/dave.toml gen_props $i
	python -m gdvb configs/dave.toml verify $i
	python -m gdvb configs/dave.toml analyze $i
done
