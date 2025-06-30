#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=defq             
#SBATCH --gpus=1                    # Is 1 GPU enough?
#SBATCH --cpus-per-gpu=4            
#SBATCH --mem-per-gpu=8G            # Allocate 8GB RAM per GPU
#SBATCH --time=01:00:00             # Job time needs to be matched to training
#SBATCH --output=earthformer_training-%j.log    # Output to a log file

module load miniconda/23
conda activate jupyter_env

REPO_DIR="$HOME/concurrent-heatwave-prediction"

source ${REPO_DIR}/config.env

mkdir -p ${TMPDIR}/data
mkdir -p ${TMPDIR}/pretrained

# copy pretrained weights
cp ${REPO_DIR}/data/pretrained/earthformer_earthnet2021.pt ${TMPDIR}/pretrained

# get stream + tas + pcha result
cp ${REPO_DIR}/data/deseason_smsub_sqrtcosw/lentis_stream250_JJA_2deg_101_deseason_smsub_sqrtcosw.nc ${TMPDIR}/data/lentis_stream.nc
cp ${REPO_DIR}/data/deseason_smsub_sqrtcosw/lentis_tas_JJA_2deg_101_deseason.nc ${TMPDIR}/data/lentis_tas.nc
cp ${REPO_DIR}/data/lat30-60/pcha_results_8a_0d.hdf5 ${TMPDIR}/data/pcha.hdf5

# copy scripts
cp ${REPO_DIR}/scripts/prediction/*.py ${TMPDIR}

cd $TMPDIR
echo "Running earthformer training script in TMPDIR: $TMPDIR"

python run.py

# copy results back to home
cp -r ${TMPDIR}/outputs ${REPO_DIR}/results
