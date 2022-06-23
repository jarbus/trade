#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --job-name=ray-serve
#SBATCH --output=serves/serve.log
#SBATCH --account=guest
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --partition=guest-gpu
#SBATCH --gres=gpu:TitanX:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
source /home/garbus/.bashrc
conda activate trade

# ===== Call your code below =====
python serve.py ${ARGS}


outpath="/home/garbus/trade/serves/CLASS_NAME/EXP_NAME/TRIAL_NAME/CHECK_NAME"
for outfile in $outpath/*.out; do
    python s2g.py "$outfile" &
done
sleep 60
