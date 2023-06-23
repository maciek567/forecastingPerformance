#!/bin/bash
#SBATCH --job-name=run_predictions
#SBATCH --time=00:05:00
#SBATCH --account=plgstockperformance-cpu
#SBATCH --partition=plgrid
 
module load python
module load spark

srun python run_predictions.py
