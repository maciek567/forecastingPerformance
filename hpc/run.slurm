#!/bin/bash
#SBATCH --job-name=run_predictions
#SBATCH --time=00:01:00
#SBATCH --account=plgstockperformance-cpu
#SBATCH --partition=plgrid
 
module load python
srun python ../predictions/run.py