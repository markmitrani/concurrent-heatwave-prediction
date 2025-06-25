#!/bin/bash
#SBATCH --job-name=jupyter_gpu
#SBATCH --partition=defq             
#SBATCH --gpus=1                    # Is 1 GPU enough?
#SBATCH --cpus-per-gpu=4            
#SBATCH --mem-per-gpu=8G            # Allocate 8GB RAM per GPU
#SBATCH --time=02:00:00             # Job time needs to be matched to training
#SBATCH --output=jupyter-%j.log     # Output to a log file

# Load environment
#module load shared
#module load 2024
#module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load miniconda/23
conda activate jupyter_env

PORT=8888
HOSTNAME=$(hostname)

# Start the Jupyter notebook server
jupyter notebook --no-browser --ip=127.0.0.1 --port=8888