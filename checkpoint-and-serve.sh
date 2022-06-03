ranger-select /work/garbus/ray_results
if [[ -f $(cat /tmp/ranger-select.txt) ]]; then
    file=$(cat /tmp/ranger-select.txt)
    exp=$(echo "$file" | grep -oP "/work/garbus/ray_results/[^/]*/\K[^/]*")
    check=$(echo "$file" | grep -oP "/work/garbus/ray_results/[^/]*/[^/]*/[^/]*/\K[^/]*")
    file=${file//\//\\\/}
    sed -e "s/CHECKPOINT_PATH/$file/"\
        -e "s/EXP_NAME/$exp/"\
        -e "s/CHECK_NAME/$check/"\
        trade_v3-serve-template.py > trade_v3-serve.py
    sbatch serve.sh
fi
