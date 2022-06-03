#class=$(ls /work/garbus/ray_results | fzf)
#exp=$(ls "/work/garbus/ray_results/$class" | fzf)
#trial=$(ls -d /work/garbus/ray_results/$class/$exp/*/ | grep -o "[^/]*/$" | fzf)
#check=$(ls -d /work/garbus/ray_results/$class/$exp/$trial*/ | grep -o "[^/]*/$" | sort -r | fzf)
#file=$(ls /work/garbus/ray_results/$class/$exp/$trial$check* | head -n 1)
ranger-select /work/garbus/ray_results
if [[ -f $(cat /tmp/ranger-select.txt) ]]; then
    file=$(cat /tmp/ranger-select.txt)
    exp=$(echo "$file" | grep -oP "/work/garbus/ray_results/[^/]*/\K[^/]*")
    echo "$file"
    echo $exp
    file=${file//\//\\\/}
    sed -e "s/CHECKPOINT_PATH/$file/" -e "s/EXP_NAME/$exp/" trade_v3-serve-template.py > trade_v3-serve.py
    #sbatch serve.sh
fi
