exp_name="$(grep -oP "name=f\"\K[^\"]*" trade_v3-tune.py)"
python slurm-launch.py --exp-name "$exp_name" --command "python trade_v3-tune.py" --load-env "trade" --partition "guest-gpu"
