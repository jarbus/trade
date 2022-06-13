#exp_name="$(grep -oP "name=f\"\K[^\"]*" tune.py)"
exp_name="nors"
python slurm-launch.py --exp-name "$exp_name" --command "python tune.py --punish" --load-env "trade" --partition "guest-gpu"
