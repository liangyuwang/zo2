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
        echo "Testing model_name: $model_name, task_id: $task_id"
        
        CMD2="python test/mezo_sgd/hf_qwen3/test_speed.py --model_name $model_name --zo_method zo2 --max_steps 30"

        OUT2="/tmp/output2_${model_name}_${task_id}.txt"

        $CMD2 2>&1 | tee $OUT2

        echo "Recording throughput..."
        
        # Count the total number of lines and determine the number of iteration lines
        total_lines2=$(wc -l < $OUT2)
        iter_lines2=$(grep -c 'Time cost after iteration' $OUT2)

        # Calculate the starting line for the last 50% of iterations
        start_line2=$(($total_lines2 - $iter_lines2 + $(($iter_lines2 / 2 + 1))))

        # Calculate average tokens per second for the last 50% of the iterations
        avg_tok_s2=$(tail -n +$start_line2 $OUT2 | grep 'tok/s' | awk '{print $8}' | awk '{total += $1; count++} END {print total/count}')

        echo -e "Model: $model_name, Task: $task_id"
        echo -e "ZO2 average throughput (last 50% iterations): ${GREEN}$avg_tok_s2 tok/s${NC}"
        
        rm $OUT2
    done
done