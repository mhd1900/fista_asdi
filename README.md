## Content:
Code for "Exoplanet detection in angular and spectral differential imaging with an accelerated proximal gradient algorithm".

## Dependencies: 
- torch (2.2.2)
- numpy (1.24.1)
- vip_hci (1.6.2)
- hciplot

## Run:
- *inject_postprocess/inject_postprocess_$algo.sh*: inject and post-process data cubes with $algo algorithm by running inject_postprocess.py with args. ($algo=pca,fista)
- *draw_roc/draw_roc_curve_$algo.sh*: draw roc curves and compute TPR and FPR for several thresholds by running draw_roc.py with args. ($algo=pca,fista)
- *test.ipynb*: run each algorithm on a test example.

## Cite: 
> Cavaco, Nicolas & Jacques, Laurent & Absil, Pierre-Antoine. (2025). Exoplanet detection in angular and spectral differential imaging with an accelerated proximal gradient algorithm. 135-140. 10.14428/esann/2025.ES2025-103. 
