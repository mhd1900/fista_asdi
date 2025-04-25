#!/bin/bash
for alpha in 0.0 2.0
do
python3 draw_roc.py "outputs_fista_$alpha/" 92 fista "40,$alpha" "None,None,2" "None,None,4" 120 "4,7" 0 1 "5,12" "empty_cubes/" 2 "outputs_fista_$alpha/"
done
