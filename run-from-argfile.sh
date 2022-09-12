if [[ $@ == *"--resume"* ]]; then
    echo "Resuming..."
    resume="--resume"
else
    resume=""
fi

dir="arg_files"
arg_file=$(ls $dir/*arg | fzf)
if [[ -f "$arg_file" ]]; then
    name=$(basename "$arg_file" .arg)
    args=$(cat "$arg_file")
    python slurm-launch.py --exp-name "$name" --command "python tune.py --name $name $args $resume" --load-env "trade" --partition "guest-gpu"
fi
