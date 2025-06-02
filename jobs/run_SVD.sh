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

REPO_DIR="$HOME/concurrent-heatwave-prediction"

# Source the shared config
source "${REPO_DIR}/config.env"

# Copy scripts to temp
cp "${REPO_DIR}${SCRIPT_AA_DIR}/${SVD_SCRIPT}" $TMPDIR/

# Copy lentis data to temp
cp "${REPO_DIR}${DATA_DIR}/${STREAM_FILE}" $TMPDIR/

cd $TMPDIR

# Run scripts
python ${SVD_SCRIPT}

# Copy back outputs to home
cp ${SVD_FILE} ${REPO_DIR}${DATA_DIR}
cp ${ZMAP_FILE} ${REPO_DIR}${DATA_DIR}
cp ${SVD_PLOT_FILE} ${REPO_DIR}${PLOTS_DIR}

echo "Finished. Output copied to ${REPO_DIR}${DATA_DIR}"