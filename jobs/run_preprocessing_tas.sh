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

# Copy script to $TMPDIR
cp "${REPO_DIR}${SCRIPT_CA_DIR}/${PRE_TAS_SCRIPT}" $TMPDIR/

# Move to $TMPDIR, run script
cd $TMPDIR
echo "Running preprocessing script in TMPDIR: $TMPDIR"

which python
python --version
python ${PRE_TAS_SCRIPT}

# Copy output file back to home directory
cp ${TAS_FILE} "${REPO_DIR}${DATA_DIR}"

echo "Finished. Output copied to "${REPO_DIR}${DATA_DIR}""
