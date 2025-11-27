# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/bin/bash
source ../scripts/set_env.sh
source ../scripts/function.sh

echo "----------------parse model_name and yaml start----------------"

# Extract model_names and yaml separately into arrays.
readarray -t model_array < <(python3 -c "
import json

with open('./st_cases.json', 'r') as f:
    data = json.load(f)

for case in data['cases']:
    print(case['model_name'])
")

readarray -t yaml_array < <(python3 -c "
import json

with open('./st_cases.json', 'r') as f:
    data = json.load(f)

for case in data['cases']:
    print(case['yaml'])
")

echo "model_names: ${model_array[@]}"
echo "yaml_array: ${yaml_array[@]}"

# Verify that the two arrays have the same length.
if [ ${#model_array[@]} -ne ${#yaml_array[@]} ]; then
    echo "ERROR: model array length is not the same as yaml array length"
    exit 1
fi

echo "----------------parse model_name and yaml finish----------------"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
MODELS_DIR="${PROJECT_ROOT}/models"

# Traverse the array
for i in "${!model_array[@]}"; do
    echo "----------------case $((i+1)): model_name=${model_array[i]}, YAML=${yaml_array[i]}----------------"
    yaml_file="${MODELS_DIR}/${model_array[i]}/config/${yaml_array[i]}"

    IPs=("${IPs_ORI[@]}")
    export YAML=$yaml_file
    export MODEL_DIR=${model_array[i]}
    launch
    wait
    check_result
    echo '----------------finish test case' $yaml_file '----------------'
done

echo '----------------all case pass----------------'