#!/bin/bash
# 遇到错误立即停止执行后续命令（按需开启）
set -e 

pkill -9 -f infer.py || true  # 加上 || true 防止没有进程可杀时脚本直接退出
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false
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
export ASCEND_LAUNCH_BLOCKING=1

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}
NPRC_PER_NODE=${NPRC_PER_NODE:-16}
NNODES=${NNODES:-1}

# 任务 1
echo "========== 开始执行任务 1: osp_14b_rl =========="
torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  infer/infer_osp.py \
  --config configs/infer/npu/osp_14b_rl.yaml 2>&1 | tee task1_rl.log

# # 任务 2
# echo "========== 开始执行任务 2: osp_14b_rl_2 =========="
# torchrun \
#   --nproc_per_node=${NPRC_PER_NODE} \
#   --nnodes=${NNODES} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   infer/infer_osp.py \
#   --config configs/infer/npu/osp_14b_rl_2.yaml 2>&1 | tee task2_rl2.log

# # 任务 3
# echo "========== 开始执行任务 3: osp_14b_rl_3 =========="
# torchrun \
#   --nproc_per_node=${NPRC_PER_NODE} \
#   --nnodes=${NNODES} \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   infer/infer_osp.py \
#   --config configs/infer/npu/osp_14b_rl_3.yaml 2>&1 | tee task3_rl3.log

echo "========== 所有任务执行完毕 =========="