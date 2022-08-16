#!/bin/bash
#SBATCH --job-name=serve-checkpoints
#SBATCH --output=/home/garbus/trade/serves/serve.log
#SBATCH --account=guest
#SBATCH --qos=low-gpu
#SBATCH --partition=guest-gpu
#SBATCH --gres=gpu:TitanX:8
#SBATCH --exclude=gpu-6-9

source DIRS.py
DIR_PATH="/work/garbus/tmp/lf-select.txt"
SERVE_PATHS="/work/garbus/tmp/most-recent-serve-dirs.txt"
echo "CHECKPOINTS TO BE SERVED:"
#cat "$DIR_PATH"
while read -r line; do 
    echo $line
    #srun --ntasks 1 --job-name=ray-serve --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 echo "begin" &
    #srun --ntasks 1 --job-name=ray-serve --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 echo hostname &
    #srun --ntasks 1 echo $line &
    #srun --ntasks 1 --job-name=ray-serve --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 hostname &
done < "$DIR_PATH"
wait
jobs
x="a"
# for dir in selected dirs
while read -r file; do
    echo "PROCESSING FILE: $file"
    if [[ -f $file ]]; then
        # get experiment information from path
        class=$(echo "$file" | grep -oP "$RESULTS_DIR/\K[^/]*")
        exp=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/\K[^/]*")
        trial=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/\K[^/]*")
        check=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/\K[^/]*")
        checkpoint_file=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*/\K[^/]*")
        check_dir_full_path=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*")
        tmp_name="/work/garbus/tmp/$(date +%s)$x"
        cp -r "$check_dir_full_path" "$tmp_name" &
        tmp_file="$tmp_name/$checkpoint_file"
        x="$x-a"
        arg_file="/home/garbus/trade/arg_files/$exp.arg"
        # run serve using arg file inferred from name
        if [[ -f "$arg_file" ]]; then
            args=$(cat "$arg_file")
            # Use srun instead of sbatch so jobs quits if shell is closed
            #echo "RUNNING COMMAND: srun --job-name=ray-serve --output=serves/serve.log --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 --ntasks-per-node=1 --cpus-per-task=16 ~/miniconda3/envs/trade/bin/python reserve.py --checkpoint $file --tmp-checkpoint $tmp_file $args &"
            srun --exclusive --ntasks 1 --job-name=ray-serve --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 ~/miniconda3/envs/trade/bin/python reserve.py --checkpoint $file --tmp-checkpoint $tmp_file $args &
            #srun --exclusive --ntasks 1 --job-name=ray-serve --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 hostname &
            #srun --exclusive hostname &

            echo "$class/$exp/$trial/$check" >> $SERVE_PATHS
            echo "Finished processing"
        else
            echo "ERROR No arg file found: $arg_file "
        fi
    else
        echo "ERROR File not found: $file"
    fi 

done < "$DIR_PATH"
jobs
wait

echo "Finished serving, creating visualizations."


while read -r dir; do
    full_dir="/home/garbus/trade/serves/$dir"
    [[ -d "$full_dir" ]] || break
    for out in $full_dir/*.out; do
        srun --job-name=vis --output=serves/vis.log --account=guest --partition=guest-compute --cpus-per-task=1 ~/miniconda3/envs/trade/bin/python s2g.py "$out" &
    done

done < $SERVE_PATHS
jobs
wait
