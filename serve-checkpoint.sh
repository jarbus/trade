source DIRS.py
/home/garbus/.local/bin/ranger-select "$RESULTS_DIR"
if [[ -f $(cat /tmp/ranger-select.txt) ]]; then
    file=$(cat /tmp/ranger-select.txt)
    class=$(echo "$file" | grep -oP "$RESULTS_DIR/\K[^/]*")
    exp=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/\K[^/]*")
    trial=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/\K[^/]*")
    check=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/\K[^/]*")
    checkpoint_file=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*/\K[^/]*")
    #file=${file//\//\\\/}
    check_dir_full_path=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/[^/]*")
    tmp_name="/work/garbus/tmp/$(date +%s)"
    echo checkpoint_file: $checkpoint_file
    echo check_dir_full_path: $check_dir_full_path
    echo tmp_name: $tmp_name
    echo cp -r "$check_dir_full_path" "$tmp_name"
    cp -r "$check_dir_full_path" "$tmp_name"
    file="$tmp_name/$checkpoint_file"
    file=${file//\//\\\/}
    echo file: $file

    sed -e "s/CHECKPOINT_PATH/$file/"\
        -e "s/EXP_NAME/$exp/"\
        -e "s/CLASS_NAME/$class/"\
        -e "s/TRIAL_NAME/$trial/"\
        -e "s/CHECK_NAME/$check/"\
        template-serve.py > serve.py
    dir="arg_files"
    #arg_file=$(ls $dir | fzf)
    arg_file="$exp.arg"
    echo $arg_file
    if [[ -f "$dir/$arg_file" ]]; then
        args=$(cat $dir/$arg_file)
        sed -e "s/\${ARGS}/$args/"\
            -e "s/CHECKPOINT_PATH/$file/"\
            -e "s/EXP_NAME/$exp/"\
            -e "s/CLASS_NAME/$class/"\
            -e "s/TRIAL_NAME/$trial/"\
            -e "s/CHECK_NAME/$check/"\
            serve-template.sh > serve.sh
        chmod +x serve.sh
        # Use srun instead of sbatch because 
        srun --job-name=ray-serve --output=serves/serve.log --account=guest --qos=low-gpu --time=24:00:00 --partition=guest-gpu --gres=gpu:TitanX:1 --ntasks-per-node=1 --cpus-per-task=16 serve.sh



        # sbatch serve.sh
        echo "$class/$exp/$trial/$check" > /work/garbus/tmp/most-recent-serve-dir
    fi
fi
