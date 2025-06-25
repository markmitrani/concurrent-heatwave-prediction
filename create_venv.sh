#!/bin/bash
#SBATCH --job-name=setup_netcdf_env
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Uses 4 cores for faster package installation
#SBATCH --time=15:00  # 15 minutes should be enough
#SBATCH --mem=4G  # 4GB should be enough for package installation

# Load module
module load miniconda/23
conda init bash
source ~/.bashrc

#conda update -n base -c defaults conda
#conda config --set channel_priority strict

# Create Conda environment (if not exists)
conda create -y --name jupyter_env --file concurrent-heatwave-prediction/requirements_conda.txt python=3.9 

# Activate the environment
conda activate jupyter_env
# Install packages from requirements files
conda install --file concurrent-heatwave-prediction/requirements_conda.txt
# py_pcha only available in pip, install this after conda reqs
pip install -r concurrent-heatwave-prediction/requirements_pip.txt --upgrade-strategy only-if-needed
# pytorch with cuda 11.8 is best installed directly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# List installed packages (sanity check)
conda list

conda deactivate