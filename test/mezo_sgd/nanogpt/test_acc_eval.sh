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
    paste <(grep 'Iteration' $OUT1) <(grep 'Iteration' $OUT2) | awk -v green="$GREEN" -v red="$RED" -v nc="$NC" '{
        split($2, loss1, ":");
        split($7, loss2, ":");
        diff_loss = loss1[2] - loss2[2];
        if (loss1[2] == loss2[2])
            printf "Iteration %s: %s✓ loss match.%s\n", $2, green, nc;
        else
            printf "Iteration %s: %s✗ Mismatch! ZO (loss): (%s), ZO2 (loss): (%s) \tLoss diff: %.6f%s\n", $2, red, loss1[2], loss2[2], diff_loss, nc;
    }'

    rm $OUT1 $OUT2
done