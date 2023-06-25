#!/bin/bash
#SBATCH --job-name=run_predictions
#SBATCH --time=00:10:00
#SBATCH --account=plgstockperformance-cpu
#SBATCH --partition=plgrid
 
module load python
module load spark

export PYTHONPATH=$PYTHONPATH:..
export PYARROW_IGNORE_TIMEZONE=1

srun python run_predictions.py
