#!/bin/bash
#
#SBATCH --job-name=example-cpu # Job name for tracking
#SBATCH --partition=cpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=1      # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem-per-cpu=15000    # RAM in MB needed per core (i.e. 1500MB * 40 threads = 60GB RAM)
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=/dcs/large/u2102661/CS310/kudu_output/random/joboutput_%j.out # Standard out from your job
#SBATCH --error=/dcs/large/u2102661/CS310/kudu_output/random/joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
source /dcs/21/u2102661/Documents/3rdYear/CS310/Classification-of-High-Tackles-in-Rugby/venv/bin/activate
srun python3 /dcs/21/u2102661/Documents/3rdYear/CS310/Classification-of-High-Tackles-in-Rugby/scripts/python/anomaly_detection.py
