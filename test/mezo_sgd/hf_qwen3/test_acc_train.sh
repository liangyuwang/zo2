#!/bin/bash

set -e
set -o pipefail

model_names=("qwen3_0_6b" "qwen3_1_7b" "qwen3_4b" "qwen3_8b" "qwen3_14b" "qwen3_32b")
task_ids=("causalLM")

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

for model_name in "${model_names[@]}"
do
    for task_id in "${task_ids[@]}"
    do
        echo "Testing model_name: $model_name"
        
        if [ "$task_id" == "causalLM" ]; then
            lr=1e-4
        else
            lr=1e-7
        fi

        CMD1="python test/mezo_sgd/hf_qwen3/test_acc.py --model_name $model_name --zo_method zo --lr $lr"
        CMD2="python test/mezo_sgd/hf_qwen3/test_acc.py --model_name $model_name --zo_method zo2 --lr $lr"

        OUT1="/tmp/output1_${model_name}_${task_id}.txt"
        OUT2="/tmp/output2_${model_name}_${task_id}.txt"

        $CMD1 2>&1 | tee $OUT1
        $CMD2 2>&1 | tee $OUT2

        echo "Comparing outputs..."
        echo -e "Model: $model_name, Task: $task_id"
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
                printf "Iteration %d: %s✗ Mismatch! ZO (loss, grad): (%s, %s), ZO2 (loss, grad): (%s, %s)\n \tLoss diff: %.6f, Proj grad diff: %.6f%s\n", $2, red, loss1[1], proj1[1], loss2[1], proj2[1], diff_loss, diff_proj, nc;
        }'

        rm $OUT1 $OUT2
    done
done