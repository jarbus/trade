#!/bin/bash

# USAGE:
# download-serves.sh [--force-gifs]

function download() {
  DIR="$1"
  notify-send "Downloading $DIR..."

  mkdir -p "/home/jack/s4/TradeEnv/trade/serves/$DIR"
  rsync -ravP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$DIR" "serves/$DIR"
  notify-send "$DIR finished downloading"
  # Fix this
  echo "dled to /home/jack/s4/TradeEnv/trade/serves/$DIR"
}



# check if first argument is not --download-only
if [ "$1" != "--download-only" ]; then
    ssh hpcc -t "source .bashrc; cd /home/garbus/trade; bash select-checkpoint.sh; bash serve-checkpoint.sh"
    scp hpcc:/work/garbus/tmp/most-recent-serve-dirs.txt /tmp/most-recent-serve-dirs.txt
    DIR="$(cat /tmp/most-recent-serve-dirs.txt)"

    while read -r dir; do
      download "$dir"
    done < /tmp/most-recent-serve-dirs.txt
    cd "/home/jack/s4/TradeEnv/trade/serves/$(fzf < /tmp/most-recent-serve-dirs.txt)"
else
    echo "Download only $2"
    ssh hpcc -t "source .bashrc; cd /home/garbus/trade; bash select-serves.sh"
    scp hpcc:/work/garbus/tmp/serves-to-download.txt /tmp/serves-to-download.txt
    cat /tmp/serves-to-download.txt
    while read -r dir; do
      d="${dir/\/home\/garbus\/trade\/serves\//}"
      parent="$(dirname $d)"
      echo "Parent: $parent\nDir: $d"
      rsync -ravP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$d" "serves/$parent"
    done < /tmp/serves-to-download.txt
    if [[ -s "/tmp/serves-to-download.txt" ]]; then
      # xargs will trim whitespace!
      cd "$(sed -r "s/garbus/jack\/s4\/TradeEnv/g" /tmp/serves-to-download.txt | fzf | xargs)"
    fi
    
    #rsync -avP --partial --info=progress2 "hpcc:/home/garbus/trade/serves/$2" serves
fi
