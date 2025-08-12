#!/bin/bash

# ====================== 需要扫描的参数 ======================
declare -A PARAM_MAP=(
    ["epochs"]="100 200"              # 对应 EPOCHS_LIST
    ["hidden"]="256 512 1024"         # 对应 HIDDENS
    ["hops"]="3 4 5 6 7 8 9 10 16 17" # 对应 HOPS_LIST
    ["fixed_splits"]="0 1"            # 对应 FIXED_SPLITS_LIST
    ["normalization"]="local_w softmax row_sum"
    ["resnet"]="0 1"
    ["layer_norm"]="0 1"
    ["att_hopwise_distinct"]="0 1"
    ["fuse_hop"]="mlp cat qkv self"
    ["online_cs"]="sub_cs sub_topk signed_cs bfs_teleport"
    ["lambda_pen"]="0.5 1.0"
    ["lambda_2hop"]="0.5 1.0"
    ["threshold"]="0.9 0.8 0.7 0.6 0.5 0.2"
    ["comm_size"]="20 30"
    ["variant"]="0 1"
    ["structure_info"]="0 1 2"
)

# ====================== 核心执行逻辑 ======================
for dataset in "cornell" "wisconsin" "texas" "film" "chameleon" "squirrel" "cora" "citeSeer" "pubmed"; do
    echo -e "\n\n======================== 开始处理数据集: $dataset ========================"
    
    # 遍历所有参数
    for param_name in "${!PARAM_MAP[@]}"; do
        # 分割参数值列表
        read -ra param_values <<< "${PARAM_MAP[$param_name]}"
        
        # 遍历每个参数值
        for param_value in "${param_values[@]}"; do
            echo -e "\n**** 正在测试参数: $param_name=$param_value ****"
            
            # 构建命令（注意参数名的下划线会自动转换为连字符）
            cmd="python train.py --dataset_name $dataset --$param_name $param_value"
            
            # 打印并执行命令
            echo "执行命令: $cmd"
            eval $cmd
            
            echo -e "**** 参数测试完成: $param_name=$param_value ****\n"
        done
    done
    
    echo -e "======================== 完成数据集: $dataset ========================\n"
done

echo "所有参数扫描完成！"