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
    
    CMD2="python test/mezo_sgd/nanogpt/test_speed.py --model_id $model_id --zo_method zo2 --max_steps 30"

    OUT2="/tmp/output2_$model_id.txt"

    $CMD2 2>&1 | tee $OUT2

    echo "Analyzing throughput..."
    
    # Count the total number of lines and determine the number of iteration lines
    total_lines2=$(wc -l < $OUT2)
    iter_lines2=$(grep -c 'Time cost after iteration' $OUT2)

    # Calculate the starting line for the last 50% of iterations
    start_line2=$(($total_lines2 - $iter_lines2 + $(($iter_lines2 / 2 + 1))))

    # Calculate average tokens per second for the last 50% of the iterations
    avg_tok_s2=$(tail -n +$start_line2 $OUT2 | grep 'tok/s' | awk '{print $8}' | awk '{total += $1; count++} END {print total/count}')

    echo -e "Model: $model_name"
    echo -e "ZO2 average throughput (last 50% iterations): ${GREEN}$avg_tok_s2 tok/s${NC}"
    
    rm $OUT2
done