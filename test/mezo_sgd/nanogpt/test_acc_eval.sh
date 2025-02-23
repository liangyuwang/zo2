#!/bin/bash

set -e
set -o pipefail

model_ids=("gpt2" "gpt2_medium" "gpt2_large" "gpt2_xl" "opt_125m" "opt_350m" "opt_1_3b" "opt_2_7b" "opt_6_7b" "opt_13b" "opt_30b" "opt_66b" "opt_175b")

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

for model_id in "${model_ids[@]}"
do
    echo "Testing model_id: $model_id"
    
    CMD1="python test/mezo_sgd/nanogpt/test_acc.py --model_id $model_id --zo_method zo --eval"
    CMD2="python test/mezo_sgd/nanogpt/test_acc.py --model_id $model_id --zo_method zo2 --eval"

    OUT1="/tmp/output1_$model_id.txt"
    OUT2="/tmp/output2_$model_id.txt"

    $CMD1 2>&1 | tee $OUT1
    $CMD2 2>&1 | tee $OUT2

    echo "Comparing outputs..."
    echo -e "Model: $model_id"
    paste <(grep 'loss' $OUT1) <(grep 'loss' $OUT2) | awk -v green="$GREEN" -v red="$RED" -v nc="$NC" '{
        split($2, loss1, ":");
        split($6, loss2, ":");
        diff_loss = loss1[2] - loss2[2];
        if (loss1[2] == loss2[2])
            printf "%s✓ Loss match.%s\n", green, nc;  # Ensure that color reset applies only to the check mark and message
        else
            printf "%s✗ Mismatch! ZO (loss): (%s), ZO2 (loss): (%s)\n \tLoss diff: %.6f%s\n", red, loss1[2], loss2[2], diff_loss, nc;
    }'

    rm $OUT1 $OUT2
done