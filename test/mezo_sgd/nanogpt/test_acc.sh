#!/bin/bash

set -e
set -o pipefail

model_ids=("gpt2" "gpt2_medium" "gpt2_large" "gpt2_xl" "opt_125m" "opt_350m" "opt_1_3b" "opt_2_7b" "opt_6_7b")

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

for model_id in "${model_ids[@]}"
do
    echo "Testing model_id: $model_id"
    
    CMD1="python test/mezo_sgd/nanogpt/test_acc.py --model_id $model_id --zo_method zo"
    CMD2="python test/mezo_sgd/nanogpt/test_acc.py --model_id $model_id --zo_method zo2"

    OUT1="/tmp/output1_$model_id.txt"
    OUT2="/tmp/output2_$model_id.txt"

    $CMD1 2>&1 | tee $OUT1
    $CMD2 2>&1 | tee $OUT2

    echo "Comparing outputs..."
    paste <(grep 'Iteration' $OUT1) <(grep 'Iteration' $OUT2) | awk -v green="$GREEN" -v red="$RED" -v nc="$NC" '{
        split($4, loss1, ",");
        split($7, proj1, ",");
        split($11, loss2, ",");
        split($14, proj2, ",");
        diff_loss = loss1[1] - loss2[1];
        diff_proj = proj1[1] - proj2[1];
        if (loss1[1] == loss2[1] && proj1[1] == proj2[1])
            printf "Iteration %d: %s✓ loss and projected grad match.%s\n", $2, green, nc;
        else
            printf "Iteration %d: %s✗ Mismatch! File1 (loss, grad): (%s, %s), File2 (loss, grad): (%s, %s)\n \tLoss diff: %.6f, Proj grad diff: %.6f%s\n", $2, red, loss1[1], proj1[1], loss2[1], proj2[1], diff_loss, diff_proj, nc;
    }'

    rm $OUT1 $OUT2
done