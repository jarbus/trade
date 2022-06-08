ranger-select /work/garbus/ray_results
if [[ -f $(cat /tmp/ranger-select.txt) ]]; then
    file=$(cat /tmp/ranger-select.txt)
    class=$(echo "$file" | grep -oP "/work/garbus/ray_results/\K[^/]*")
    exp=$(echo "$file" | grep -oP "/work/garbus/ray_results/[^/]*/\K[^/]*")
    check=$(echo "$file" | grep -oP "/work/garbus/ray_results/[^/]*/[^/]*/[^/]*/\K[^/]*")
    file=${file//\//\\\/}
    sed -e "s/CHECKPOINT_PATH/$file/"\
        -e "s/EXP_NAME/$exp/"\
        -e "s/CLASS_NAME/$class/"\
        -e "s/CHECK_NAME/$check/"\
        template-serve.py > serve.py
    sbatch serve.sh
fi
