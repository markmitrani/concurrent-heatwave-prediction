#!/bin/bash
#SBATCH --job-name=lentis_preprocessing
#SBATCH --partition=defq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=preprocessing-%j.log

# Load environment
module load miniconda
conda activate netcdf_env

REPO_DIR="$HOME/concurrent-heatwave-prediction"

# Source shared config
source "${REPO_DIR}/config.env"

# Copy script to $TMPDIR
cp "${REPO_DIR}${SCRIPT_PREP_DIR}/${PRE_FULL_SCRIPT}" $TMPDIR/

# Move to $TMPDIR, run script
cd $TMPDIR
echo "Running preprocessing script in TMPDIR: $TMPDIR"
ls

python ${PRE_FULL_SCRIPT} --stream

# Copy the output file(s) back to home directory
cp ${FULL_STREAM_FILE} "${REPO_DIR}/data"
# cp ${FULL_OLR_FILE} "${REPO_DIR}${DATA_DIR}"