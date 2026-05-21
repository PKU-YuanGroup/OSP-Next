echo "start process..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WANDB_MODE="offline"
export WANDB_API_KEY="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"
wandb login --relogin $WANDB_API_KEY
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

export TOKENIZERS_PARALLELISM=false

export ASCEND_LAUNCH_BLOCKING=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600

# 每个节点的卡数
NPROC_PER_NODE=${MA_NUM_GPUS:-16}

# 节点总数
NNODES=${MA_NUM_HOSTS:-8}

# 当前节点的 rank
NODE_RANK=${VC_TASK_INDEX:-0}

# Master 地址：取 VC_WORKER_HOSTS 第一个节点的域名，通过 DNS 解析成 IP
MASTER_HOSTNAME=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)
MASTER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname('${MASTER_HOSTNAME}'))")

# 也可以直接用 MA_CURRENT_IP（仅当 NODE_RANK=0 的节点 IP 作为 master）
# MASTER_ADDR 这里不能用 MA_CURRENT_IP，因为每个节点的 MA_CURRENT_IP 是自己的 IP

MASTER_PORT=${MASTER_PORT:-29501}
WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NNODES=${NNODES}"
echo "NODE_RANK=${NODE_RANK}"
echo "WORLD_SIZE=${WORLD_SIZE}"

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --nnodes=${NNODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_osp_RL_lora.py \
  --config configs/train/npu/osp_14b_RL.yaml