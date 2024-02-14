#!/bin/bash
#
#SBATCH --job-name=example-gpu # Job name for tracking
#SBATCH --partition=falcon     # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12     # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=kudu_jobs/joboutput_%j.out # Standard out from your job
#SBATCH --error=kudu_jobs/joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Executing the pose estimation prediction ##
# module load CUDA

## This is specific to my environment, you will need to change this to your own ##
source /dcs/21/u2102661/Documents/3rdYear/CS310/Classification-of-High-Tackles-in-Rugby/venv/bin/activate
srun python3 /dcs/21/u2102661/Documents/3rdYear/CS310/Classification-of-High-Tackles-in-Rugby/scripts/pose_estimation.py