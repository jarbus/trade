source DIRS.py
ranger-select "$RESULTS_DIR"
if [[ -f $(cat /tmp/ranger-select.txt) ]]; then
    file=$(cat /tmp/ranger-select.txt)
    class=$(echo "$file" | grep -oP "$RESULTS_DIR/\K[^/]*")
    exp=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/\K[^/]*")
    trial=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/\K[^/]*")
    check=$(echo "$file" | grep -oP "$RESULTS_DIR/[^/]*/[^/]*/[^/]*/\K[^/]*")
    file=${file//\//\\\/}
    sed -e "s/CHECKPOINT_PATH/$file/"\
        -e "s/EXP_NAME/$exp/"\
        -e "s/CLASS_NAME/$class/"\
        -e "s/TRIAL_NAME/$trial/"\
        -e "s/CHECK_NAME/$check/"\
        template-serve.py > serve.py
    dir="arg_files"
    arg_file=$(ls $dir | fzf)
    if [[ -f "$dir/$arg_file" ]]; then
        args=$(cat $dir/$arg_file)
        sed -e "s/\${ARGS}/$args/"\
            serve-template.sh > serve.sh
        sbatch serve.sh
    fi
fi
