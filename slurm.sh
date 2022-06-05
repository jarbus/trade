exp_name="$(grep -oP "name=f\"\K[^\"]*" tune.py)"
python slurm-launch.py --exp-name "$exp_name" --command "python tune.py" --load-env "trade" --partition "guest-gpu"
