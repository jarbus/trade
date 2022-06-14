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

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
source /home/garbus/.bashrc
conda activate trade

# ===== Call your code below =====
python serve.py --class-name meeting --random-start --dist-coeff 0.2 --survival-bonus 1.0 --batch-size 1000 --death-prob 0.1 --num-steps 10000000 --checkpoint-interval 500 --twonn-coeff 0.1 --health-baseline
