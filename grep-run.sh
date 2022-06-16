ranger-select
file=$(cat /tmp/ranger-select.txt)
grep -P "Result for .*|mut_exchange_total_mean.*" "$file"  > "$file".extracted
