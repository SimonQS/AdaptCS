#!/usr/bin/env bash
# 跑所有的 baselines

# set -euo pipefail     # ① 出错即停，未定义变量报错，管道出错即停

# ====================== 需要扫描的参数 ======================
# ② 正确写法：先声明，再为每个键赋“空格分隔”的取值列表
declare -A PARAM_MAP         

PARAM_MAP["model"]="icsgnn qdgnn coclep \
icsgnn_tanh qdgnn_tanh coclep_tanh \
icsgnn_acm qdgnn_acm coclep_acm \
acmsmn"
# k_core k_truss clique acmsmn"

# ====================== 其他配置 ======================
date_str=$(date +%Y_%m_%d)
path_name="results_csv/batch_results_${date_str}.csv"
mkdir -p "$(dirname "$path_name")"   # ③ 若目录不存在自动创建

datasets=(
  "cornell" "wisconsin" "texas" "film"
  "chameleon" "squirrel"
  "cora" "citeseer" "pubmed" "reddit"
)
# datasets=("squirrel" "cora" "citeseer" "pubmed" "reddit")

# ====================== 核心执行逻辑 ======================
for dataset in "${datasets[@]}"; do
  echo -e "\n\n======================== 开始处理数据集: $dataset ========================"

  for param_name in "${!PARAM_MAP[@]}"; do
    # ④ 将 value 字符串拆成数组；IFS 默认空白即可
    read -r -a param_values <<< "${PARAM_MAP[$param_name]}"

    for param_value in "${param_values[@]}"; do
      echo -e "\n**** 正在测试参数: $param_name=$param_value ****"

      # ⑤ 如需把下划线换成连字符可用下行；若不需要可直接 "--${param_name}"
      param_flag="--${param_name//_/-}"

      # ⑥ 用数组构造命令，避免引号转义麻烦
      cmd=(python train.py --batch_csv "$path_name" --dataset_name "$dataset" "$param_flag" "$param_value")

      echo "执行命令: ${cmd[*]}"
      "${cmd[@]}"

      echo -e "**** 参数测试完成: $param_name=$param_value ****\n"
    done
  done

  echo -e "======================== 完成数据集: $dataset ========================\n"
done

echo "所有参数扫描完成！"
