#!/bin/bash
for flux in 0.0 0.25 0.5
do
for seed in 4 5 6
do
python3 main.py "outputs_pca/" 1 pca "30,0" "None,None,2" "None,None,4" 120 "4,7" $flux $seed "5,12" "empty_cubes/" 2
done
done
