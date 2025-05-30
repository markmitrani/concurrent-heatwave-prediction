#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

# Load environment
module load miniconda
conda activate netcdf_env

REPO_DIR="~/concurrent-heatwave-prediction"

# Source the shared config
source "${REPO_DIR}/config.env"

# Copy script to temp
cp "${REPO_DIR}${SCRIPT_AA_DIR}/${AA_SCRIPT}" $TMPDIR

# Copy previously saved svd.hdf5 to temp
cp "${REPO_DIR}${DATA_DIR}${SVD_FILE}" $TMPDIR

# Move to temp space
cd $TMPDIR

# Run script
python ${AA_SCRIPT}

# Copy back outputs to home
cp ${PCHA_FILE} ${REPO_DIR}${DATA_DIR}

echo "Finished. Output copied to ${REPO_DIR}${DATA_DIR}"