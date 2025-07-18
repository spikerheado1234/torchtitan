#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
#NGPU=${NGPU:-"1"}
#LOG_RANK=${LOG_RANK:-0}
NGPU=2
LOG_RANK=0
## This is for debugging only. ##
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
## This is for 8b evals. ##
#CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
