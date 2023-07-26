#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --output=/home/garbus/evotrade/run-batch-vis.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# we need to use this to prevent dirs with whitespaces from expanding into two dirs
find "$1" -name "*.out" -print | while IFS= read -r file; do
    python /home/garbus/trade/evotrade-s2g.py "$file" &
done
wait
