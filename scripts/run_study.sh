#!/bin/bash

echo 'Executing the First Study: MNIST_Conv_Big'
for i in {10..14}
do
	python -m gdvb configs/mcb.toml E --seed $i
done

echo 'Executing the Second Study: DAVE-2'
for i in {10..14}
do
	python -m gdvb configs/dave.toml E --seed $i
done
