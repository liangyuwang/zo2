#!/bin/bash

set -e
set -o pipefail

model_names=("opt_125m" "opt_350m" "opt_1_3b" "opt_2_7b" "opt_6_7b")
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
        
        CMD1="python test/mezo_sgd/hf_opt/test_speed.py --model_name $model_name --task $task_id --zo_method zo --max_steps 30"
        CMD2="python test/mezo_sgd/hf_opt/test_speed.py --model_name $model_name --task $task_id --zo_method zo2 --max_steps 30"

        OUT1="/tmp/output1_${model_name}_${task_id}.txt"
        OUT2="/tmp/output2_${model_name}_${task_id}.txt"

        $CMD1 2>&1 | tee $OUT1
        $CMD2 2>&1 | tee $OUT2

        echo "Analyzing throughput..."
        
        # Count the total number of lines and determine the number of iteration lines
        total_lines1=$(wc -l < $OUT1)
        total_lines2=$(wc -l < $OUT2)
        iter_lines1=$(grep -c 'Time cost after iteration' $OUT1)
        iter_lines2=$(grep -c 'Time cost after iteration' $OUT2)

        # Calculate the starting line for the last 50% of iterations
        start_line1=$(($total_lines1 - $iter_lines1 + $(($iter_lines1 / 2 + 1))))
        start_line2=$(($total_lines2 - $iter_lines2 + $(($iter_lines2 / 2 + 1))))

        # Calculate average tokens per second for the last 50% of the iterations
        avg_tok_s1=$(tail -n +$start_line1 $OUT1 | grep 'tok/s' | awk '{print $8}' | awk '{total += $1; count++} END {print total/count}')
        avg_tok_s2=$(tail -n +$start_line2 $OUT2 | grep 'tok/s' | awk '{print $8}' | awk '{total += $1; count++} END {print total/count}')

        ratio=$(echo "scale=2; $avg_tok_s2 / $avg_tok_s1 * 100" | bc)

        echo -e "Model: $model_name, Task: $task_id"
        echo -e "ZO average throughput (last 50% iterations): ${GREEN}$avg_tok_s1 tok/s${NC}"
        echo -e "ZO2 average throughput (last 50% iterations): ${GREEN}$avg_tok_s2 tok/s${NC}"
        echo -e "Throughput ratio of ZO2 to ZO (last 50% iterations): ${GREEN}$ratio%${NC}"

        rm $OUT1 $OUT2
    done
done