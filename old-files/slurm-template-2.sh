#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!

#SBATCH --mail-type=END
#SBATCH --mail-user=garbus@brandeis.edu
#SBATCH --job-name=runs/${JOB_NAME}
#SBATCH --output=runs/${JOB_NAME}.log
#SBATCH --account=guest
#SBATCH --qos=low-gpu
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --partition=guest-gpu
#SBATCH --gres=gpu:TitanX:8
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1GB

# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
source /home/garbus/.bashrc
conda activate trade

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
#redis_password=$(uuidgen)
redis_password="longredispassword"
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
      IFS=' ' read -ra ADDR <<< "$ip"
      if [[ ${#ADDR[0]} -gt 16 ]]; then
          ip=${ADDR[1]}
      else
          ip=${ADDR[0]}
      fi
      echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6380
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" --gres=gpu:TitanX:8 \
    ray start --head --node-ip-address="$ip" \
         --port=$port \
         --redis-password="$redis_password"\
         --node-manager-port=6800 \
         --object-manager-port=6801 \
         --ray-client-server-port=20001 \
         --redis-shard-ports=6802 \
         --min-worker-port=20002 \
         --max-worker-port=29999 \
         --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "STARTING WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --gres=gpu:TitanX:8 \
        ray start --address "$ip_head" \
            --redis-password="$redis_password"\
            --node-manager-port=6800 \
            --object-manager-port=6801 \
            --min-worker-port=20002 \
            --max-worker-port=29999 \
            --block &



done
sleep 30


# ===== Call your code below =====
${COMMAND_PLACEHOLDER} --second-cluster
