#!/bin/bash

# USAGE:
# download-serves.sh [--force-gifs]

function download() {
  DIR="$1"
  notify-send "Downloading $DIR..."
  rsync -avP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$CLASS_DIR" serves
  notify-send "$DIR finished downloading"
  # Fix this
  echo "dled to /home/jack/s4/TradeEnv/trade/serves/$DIR"
}



# check if first argument is not --download-only
if [ "$1" != "--download-only" ]; then
    ssh hpcc -t "source .bashrc; cd /home/garbus/trade; bash serve-checkpoint.sh"
    scp hpcc:/work/garbus/tmp/most-recent-serve-dirs.txt /tmp/most-recent-serve-dirs.txt
    DIR=$(cat /tmp/most-recent-serve-dirs.txt)
    CLASS_DIR=$(echo "$DIR" | grep -o '^[^/]*')

    while read -r dir; do
      download "$dir"
    done < /tmp/most-recent-serve-dirs.txt
else
    echo "Download only $2"
    rsync -avP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$2" serves
fi







#cd serves/$DIR
