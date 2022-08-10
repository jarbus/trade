source /home/garbus/.bashrc
conda activate trade
DIR_PATH="/tmp/lf-select.txt"
SERVE_PATHS="/work/garbus/tmp/most-recent-serve-dirs.txt"
rm -f $SERVE_PATHS
echo "" > $DIR_PATH
source DIRS.py
/home/garbus/.local/bin/lf -selection-path $DIR_PATH /work/garbus/ray_results
echo " " >> $DIR_PATH
# for dir in selected dirs
while read -r file; do
    echo "$file"
    if [[ -f $file ]]; then
        # get experiment information from path
        class=$(echo "$file" | grep -oP "$RESULTS_DIR/\K[^/]*")
        exp=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/\K[^/]*")
        trial=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/\K[^/]*")
        check=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/\K[^/]*")
        checkpoint_file=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*/\K[^/]*")
        #file=${file//\//\\\/}
        check_dir_full_path=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*")
        tmp_name="/work/garbus/tmp/$(date +%s)"
        cp -r "$check_dir_full_path" "$tmp_name"
        tmp_file="$tmp_name/$checkpoint_file"
        echo "file: $file"
        #file=${file//\//\\\/}
        arg_file="/home/garbus/trade/arg_files/$exp.arg"
        if [[ -f "$arg_file" ]]; then
            args=$(cat "$arg_file")
                #serve-template.sh > serve.sh
            chmod +x serve.sh
            # Use srun instead of sbatch because 

            echo "srun --job-name=ray-serve --output=serves/serve.log --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 --ntasks-per-node=1 --cpus-per-task=16 python reserve.py --checkpoint $file --tmp-checkpoint $tmp_file $args"

            echo "srun --job-name=ray-serve --output=serves/serve.log --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 --ntasks-per-node=1 --cpus-per-task=16 ~/miniconda3/envs/trade/bin/python reserve.py --checkpoint $file --tmp-checkpoint $tmp_file $args &"
            srun --job-name=ray-serve --output=serves/serve.log --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 --ntasks-per-node=1 --cpus-per-task=16 ~/miniconda3/envs/trade/bin/python reserve.py --checkpoint $file --tmp-checkpoint $tmp_file $args &


            # sbatch serve.sh
            echo "$class/$exp/$trial/$check" >> $SERVE_PATHS
        else
            echo "No arg file found"
        fi
    else
        echo "File not found: $file"
    fi 


done < $DIR_PATH

wait

while read -r dir; do
    full_dir="/home/garbus/trade/serves/$dir"
    [[ -d "$full_dir" ]] || break
    for out in $full_dir/*.out; do
        srun --job-name=vis --output=serves/vis.log --account=guest --partition=guest-compute --cpus-per-task=1 ~/miniconda3/envs/trade/bin/python s2g.py "$out" &
    done

done < $SERVE_PATHS
wait
