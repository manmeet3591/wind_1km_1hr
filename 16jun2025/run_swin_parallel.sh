#!/bin/bash

MASTER_ADDR=$1
NNODES=$2
NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}

# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=180
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_P2P_LEVEL=SYS
# export NCCL_P2P_DISABLE=0

# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=180
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_P2P_LEVEL=SYS
# export NCCL_IB_DISABLE=1    # (optional; use if InfiniBand errors)



export MASTER_PORT=$((10000 + $RANDOM % 10000))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"

# apptainer exec \
#   --bind /scratch/08105/ms86336:/opt/notebooks \
#   --nv apptainer_multi_gpu.sif \
#   torchrun \
#     --nproc_per_node=8 --nnodes=5 --rdzv_backend=c10d --rdzv_endpoint=host:port \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     /scratch/08105/ms86336/wind_1km_1hr/swin_transformer_wind_operational_parallel.py
# apptainer exec \
#   --bind /scratch/08105/ms86336:/opt/notebooks \
#   --nv apptainer_multi_gpu.sif \
# torchrun \
#   --nproc_per_node=1 \
#   --nnodes=$NNODES \
#   --node_rank=$NODE_RANK \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#   /scratch/08105/ms86336/wind_1km_1hr/swin_transformer_wind_operational_parallel.py

apptainer exec \
  --bind /scratch/08105/ms86336:/opt/notebooks \
  --nv apptainer_multi_gpu.sif \
  torchrun \
    --nproc_per_node=1 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    /scratch/08105/ms86336/wind_1km_1hr/swin_transformer_wind_operational_parallel.py




