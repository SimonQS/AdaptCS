#!/bin/bash

# ====================== 需要扫描的参数 ======================
declare -A PARAM_MAP=(
    # ["hidden"]="64 128 256 512 1024"         # 对应 HIDDENS
    # ["hops"]="3 4 5 6 7 8 9" # 对应 HOPS_LIST
    # ["fixed_splits"]="0 1"            # 对应 FIXED_SPLITS_LIST
    # ["svd_rank"]="16 32 50 64 100 128"  
    # ["normalization"]="local_w global_w none"
    # ["resnet"]="0 1"
    # ["layer_norm"]="0 1"
    # ["att_hopwise_distinct"]="0 1"
    # ["online_cs"]="adaptive_cs bfs_teleport"
    # ["online_cs"]="bfs_teleport"
    # ["threshold"]="0.9 0.8 0.6 0.5 0.2"
    # ["top"]="2 3 4 5 6"
    # ["dropout"]="0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"
    # ["structure_info"]="0 1 2"
    # ["approach"]="distinct_hop distinct_hop_svds_low"
    ["approach"]="distinct_hop_svds_low"
)

# ====================== 核心执行逻辑 ======================
date_str=$(date +%Y_%m_%d)
path_name="results_csv/batch_results_${date_str}.csv"
# for dataset in "film" "chameleon" "squirrel" ; do
for dataset in "cornell" "wisconsin" "texas" "film" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "reddit"; do
# for dataset in "reddit"; do
    echo -e "\n\n======================== 开始处理数据集: $dataset ========================"

    for param_name in "${!PARAM_MAP[@]}"; do
        read -ra param_values <<< "${PARAM_MAP[$param_name]}"

        for param_value in "${param_values[@]}"; do
            echo -e "\n**** 正在测试参数: $param_name=$param_value ****"

            # 将 param_name 中的下划线替换为连字符
            param_flag="--${param_name}"

            # 构建并执行命令
            cmd="python train.py --batch_csv $path_name --dataset_name $dataset $param_flag $param_value"
            echo "执行命令: $cmd"
            eval $cmd

            echo -e "**** 参数测试完成: $param_name=$param_value ****\n"
        done
    done

    echo -e "======================== 完成数据集: $dataset ========================\n"
done

echo "所有参数扫描完成！"