#!/bin/bash
jid1=$(sbatch jobs/run_preprocessing.sh)
jid2=$(sbatch --dependency=afterok:$jid1 jobs/run_SVD.sh)
jid3=$(sbatch --dependency=afterok:$jid2 jobs/run_AA.sh)