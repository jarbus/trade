#!/bin/bash

# USAGE:
# download-serves.sh [--force-gifs]
ssh hpcc -t "source .bashrc; cd /home/garbus/trade; bash serve-checkpoint.sh"
DIR=$(ssh hpcc cat /work/garbus/tmp/most-recent-serve-dir)

CLASS_DIR=$(echo $DIR | grep -o '^[^/]*')
notify-send "Downloading $DIR..."
rsync -avP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$CLASS_DIR" serves
notify-send "$DIR finished downloading"
cd serves/$DIR
