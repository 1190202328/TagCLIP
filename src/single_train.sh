#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$((RANDOM + 10000)) \
  src/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3}
