dir="arg_files"
arg_file=$(ls $dir | fzf)
if [[ -f "$dir/$arg_file" ]]; then
    name=$(basename "$arg_file" .arg)
    args=$(cat "$dir/$arg_file")
    python slurm-launch.py --exp-name "$name" --command "python tune.py --name $name $args" --load-env "trade" --partition "guest-gpu"
fi
