#!/bin/bash
DIR_PATH="/work/garbus/tmp/serves-to-download.txt"
source /home/garbus/.bashrc
conda activate trade
echo "" > $DIR_PATH
source DIRS.py
/home/garbus/.local/bin/lf -selection-path "$DIR_PATH" /home/garbus/trade/serves
echo " " >> $DIR_PATH 
