#!/bin/bash
DIR_PATH="/work/garbus/tmp/lf-select.txt"
SERVE_PATHS="/work/garbus/tmp/most-recent-serve-dirs.txt"
source /home/garbus/.bashrc
conda activate trade
rm -f $SERVE_PATHS
echo "" > $DIR_PATH
source DIRS.py
/home/garbus/.local/bin/lf -selection-path "$DIR_PATH" /work/garbus/ray_results
echo " " >> $DIR_PATH 
