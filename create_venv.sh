#!/bin/bash
#SBATCH --job-name=setup_netcdf_env
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Uses 4 cores for faster package installation
#SBATCH --time=15:00  # 15 minutes should be enough
#SBATCH --mem=4G  # 4GB should be enough for package installation

# Load module
module load miniconda

#conda update -n base -c defaults conda
#conda config --set channel_priority strict
conda config --set solver libmamba

# Create Conda environment (if not exists)
conda create -y --name netcdf_env --file requirements_conda.txt python=3.9 

# Activate the environment
conda activate netcdf_env
# Install packages from requirements files
#conda install --file requirements_conda.txt --solver=libmamba
# py_pcha only available in pip, install this after conda reqs
pip install -r requirements_pip.txt --upgrade-strategy only-if-needed

# List installed packages (sanity check)
conda list

conda deactivate