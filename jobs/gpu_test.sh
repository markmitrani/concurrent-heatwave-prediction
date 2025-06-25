#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=defq             
#SBATCH --gpus=1                    # Is 1 GPU enough?
#SBATCH --cpus-per-gpu=4            
#SBATCH --mem-per-gpu=8G            # Allocate 8GB RAM per GPU
#SBATCH --time=00:15:00             # Job time needs to be matched to training
#SBATCH --output=gpu_test-%j.log     # Output to a log file

module load miniconda/23
conda activate jupyter_env

REPO_DIR="$HOME/concurrent-heatwave-prediction"

# Source the shared config
source "${REPO_DIR}/config.env"

nvidia-smi
python ${REPO_DIR}${SCRIPT_PRED_DIR}/cuda_test.py