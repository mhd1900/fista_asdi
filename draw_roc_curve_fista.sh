#!/bin/bash
for alpha in 2.0
do
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name="draw_roc"
#SBATCH --time=00:20:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16000 # megabytes
#SBATCH --partition=batch
#
#
#SBATCH --output=draw_roc.txt

module purge
ml releases/2023b
ml Python/3.11.5-GCCcore-13.2.0
source ~/.venv/bin/activate
srun python3 draw_roc.py "outputs_fista_grid_24_40_$alpha/" 92 fista "40,$alpha" "None,None,2" "None,None,4" 120 "4,7" 0 1 "5,12" "empty_cubes/" 2 "outputs_fista_grid_24_40_$alpha/"
deactivate
EOT
done
