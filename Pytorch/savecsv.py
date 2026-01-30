<<<<<<< HEAD
# import re
# import csv
# import os
# from glob import glob

# # Folder path for logs
# log_folder_path = "logs/"

# # Updated regular expression to extract all parameters from the log line.
# # This regex assumes log lines of the form:
# #
# # 2025-02-17 10:29:55: Run 9 Summary - model=acmsmn, dataset=texas, variant=1, init_layers_X=2, structure_info=2, Hidden=1024, lr=0.0100, weight_decay=0.001000, dropout=0.2000, hops=9, resnet=0, layer_norm=0, att_hopwise_distinct=0, fuse_hop=mlp, layers=1, Test_Mean=0.9197, Test_Std=0.0248, runtime_average=17.02s, epoch_average=171.94ms, normalization=local_w, online_cs=sub_cs, lambda_pen=1.0, lambda_2hop=0.5, threshold=0.9, comm_size=20, CommunitySearchTime=0.027379s, NMI=0.0000 (std=0.0000), Jaccard=0.9367 (std=0.0283), F1=0.9671 (std=0.0141)
# #
# log_pattern = re.compile(
#     r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Run (?P<run>\d+) Summary - "
#     r"model=(?P<model>\w+), dataset=(?P<dataset>\w+), variant=(?P<variant>[\d.]+), "
#     r"init_layers_X=(?P<init_layers_X>\d+), structure_info=(?P<structure_info>\d+), "
#     r"Hidden=(?P<hidden>\d+), lr=(?P<lr>[\d.]+), weight_decay=(?P<weight_decay>[\d.]+), "
#     r"dropout=(?P<dropout>[\d.]+), hops=(?P<hops>\d+), resnet=(?P<resnet>\d+), "
#     r"layer_norm=(?P<layer_norm>\d+), att_hopwise_distinct=(?P<att_hopwise_distinct>\d+), "
#     r"fuse_hop=(?P<fuse_hop>\w+), layers=(?P<layers>\d+), "
#     r"Test_Mean=(?P<test_mean>[\d.]+), Test_Std=(?P<test_std>[\d.]+), "
#     r"runtime_average=(?P<runtime_avg>[\d.]+)s, epoch_average=(?P<epoch_avg>[\d.]+)ms, "
#     r"normalization=(?P<normalization>\w+), online_cs=(?P<online_cs>\w+), "
#     r"lambda_pen=(?P<lambda_pen>[\d.]+), lambda_2hop=(?P<lambda_2hop>[\d.]+), "
#     r"threshold=(?P<threshold>[\d.]+), comm_size=(?P<comm_size>\d+), "
#     r"CommunitySearchTime=(?P<cs_time>[\d.]+)s, "
#     r"NMI=(?P<cs_nmi>[\d.]+) \(std=(?P<cs_nmi_std>[\d.]+)\), "
#     r"Jaccard=(?P<cs_jaccard>[\d.]+) \(std=(?P<cs_jaccard_std>[\d.]+)\), "
#     r"F1=(?P<cs_f1>[\d.]+) \(std=(?P<cs_f1_std>[\d.]+)\)"
# )

# def process_logs_for_date(date):
#     # Format the CSV file path to include the date
#     csv_file_path = f"results_csv/acm_torch_logs_{date}.csv"
#     os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         # Write the header row with all parameters
#         writer.writerow([
#             "Timestamp", "Run", "Model", "Dataset", "Variant", "Init Layers X",
#             "Structure Info", "Hidden", "lr", "weight_decay", "dropout", "hops",
#             "resnet", "layer_norm", "att_hopwise_distinct", "fuse_hop", "layers",
#             "Test Mean", "Test Std", "runtime_average", "epoch_average", "normalization",
#             "online_cs", "lambda_pen", "lambda_2hop", "threshold", "comm_size",
#             "CommunitySearchTime", "NMI", "NMI_std", "Jaccard", "Jaccard_std", "F1", "F1_std"
#         ])

#         # Iterate through all log files in the logs folder that contain the given date in the filename
#         for log_file_path in glob(os.path.join(log_folder_path, f"*{date}*.log")):
#             print(f"Processing file: {log_file_path}")
#             with open(log_file_path, 'r') as log_file:
#                 for line in log_file:
#                     match = log_pattern.match(line)
#                     if match:
#                         writer.writerow([
#                             match.group("timestamp"),
#                             match.group("run"),
#                             match.group("model"),
#                             match.group("dataset"),
#                             match.group("variant"),
#                             match.group("init_layers_X"),
#                             match.group("structure_info"),
#                             match.group("hidden"),
#                             match.group("lr"),
#                             match.group("weight_decay"),
#                             match.group("dropout"),
#                             match.group("hops"),
#                             match.group("resnet"),
#                             match.group("layer_norm"),
#                             match.group("att_hopwise_distinct"),
#                             match.group("fuse_hop"),
#                             match.group("layers"),
#                             match.group("test_mean"),
#                             match.group("test_std"),
#                             match.group("runtime_avg"),
#                             match.group("epoch_avg"),
#                             match.group("normalization"),
#                             match.group("online_cs"),
#                             match.group("lambda_pen"),
#                             match.group("lambda_2hop"),
#                             match.group("threshold"),
#                             match.group("comm_size"),
#                             match.group("cs_time"),
#                             match.group("cs_nmi"),
#                             match.group("cs_nmi_std"),
#                             match.group("cs_jaccard"),
#                             match.group("cs_jaccard_std"),
#                             match.group("cs_f1"),
#                             match.group("cs_f1_std")
#                         ])
#                         print(f"Matched and wrote: {match.groups()}")
#                     else:
#                         print(f"No match for line: {line}")
#     print(f"Logs have been successfully saved to {csv_file_path}.")

# # Example usage: Process logs for a specific date (adjust date as needed)
# process_logs_for_date("2025_02_17")

import re
import csv
import os
from glob import glob
from datetime import datetime, timedelta

# Folder path for logs
log_folder_path = "logs/"

log_pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Run (?P<run>\d+) Summary - "
    r"model=(?P<model>\w+), dataset=(?P<dataset>\w+), variant=(?P<variant>[\d.]+), "
    r"init_layers_X=(?P<init_layers_X>\d+), structure_info=(?P<structure_info>\d+), "
    r"Hidden=(?P<hidden>\d+), lr=(?P<lr>[\d.]+), weight_decay=(?P<weight_decay>[\d.]+), "
    r"dropout=(?P<dropout>[\d.]+), hops=(?P<hops>\d+), resnet=(?P<resnet>\d+), "
    r"layer_norm=(?P<layer_norm>\d+), att_hopwise_distinct=(?P<att_hopwise_distinct>\d+), "
    r"fuse_hop=(?P<fuse_hop>\w+), layers=(?P<layers>\d+), "
    r"Test_Mean=(?P<test_mean>[\d.]+), Test_Std=(?P<test_std>[\d.]+), "
    r"runtime_average=(?P<runtime_avg>[\d.]+)s, epoch_average=(?P<epoch_avg>[\d.]+)ms, "
    r"normalization=(?P<normalization>\w+), online_cs=(?P<online_cs>\w+), "
    r"lambda_pen=(?P<lambda_pen>[\d.]+), lambda_2hop=(?P<lambda_2hop>[\d.]+), "
    r"threshold=(?P<threshold>[\d.]+), comm_size=(?P<comm_size>\d+), "
    r"CommunitySearchTime=(?P<cs_time>[\d.]+)s, "
    r"NMI=(?P<cs_nmi>[\d.]+) \(std=(?P<cs_nmi_std>[\d.]+)\), "
    r"Jaccard=(?P<cs_jaccard>[\d.]+) \(std=(?P<cs_jaccard_std>[\d.]+)\), "
    r"F1=(?P<cs_f1>[\d.]+) \(std=(?P<cs_f1_std>[\d.]+)\)"
)

def parse_date(date_str, input_format="%Y_%m_d"):
    """将字符串日期转换为datetime对象"""
    return datetime.strptime(date_str, input_format)

def date_range(start_date, end_date):
    """生成日期范围内所有日期的生成器"""
    delta = end_date - start_date
    for i in range(delta.days + 1):
        yield start_date + timedelta(days=i)

def process_logs_for_date_range(start_date_str, end_date_str):
    """
    处理指定日期范围内的所有日志文件
    :param start_date_str: 开始日期字符串 (格式: "YYYY_MM_DD")
    :param end_date_str: 结束日期字符串 (格式: "YYYY_MM_DD")
    """
    # 转换输入日期
    start_date = parse_date(start_date_str, "%Y_%m_%d")
    end_date = parse_date(end_date_str, "%Y_%m_%d")
    
    # 创建主CSV文件
    master_csv_path = f"results_csv/master_logs_{start_date_str}_to_{end_date_str}.csv"
    os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)
    
    with open(master_csv_path, mode='w', newline='') as master_csv:
        writer = csv.writer(master_csv)
        # 只写一次表头
        writer.writerow([
            "Timestamp", "Run", "Model", "Dataset", "Variant", "Init Layers X",
            "Structure Info", "Hidden", "lr", "weight_decay", "dropout", "hops",
            "resnet", "layer_norm", "att_hopwise_distinct", "fuse_hop", "layers",
            "Test Mean", "Test Std", "runtime_average", "epoch_average", "normalization",
            "online_cs", "lambda_pen", "lambda_2hop", "threshold", "svd_rank", "top", "comm_size",
            "CommunitySearchTime", "NMI", "NMI_std", "Jaccard", "Jaccard_std", "F1", "F1_std"
        ])

        # 遍历日期范围内的每一天
        for single_date in date_range(start_date, end_date):
            date_str = single_date.strftime("%Y_%m_%d")
            print(f"\nProcessing date: {date_str}")
            
            # 查找匹配的日志文件
            log_files = glob(os.path.join(log_folder_path, f"*{date_str}*.log"))
            
            if not log_files:
                print(f"No log files found for date {date_str}")
                continue
                
            # 处理每个日志文件
            for log_file_path in log_files:
                print(f"Processing file: {log_file_path}")
                with open(log_file_path, 'r') as log_file:
                    for line in log_file:
                        match = log_pattern.match(line)
                        if match:
                            writer.writerow([
                                match.group("timestamp"),
                                match.group("run"),
                                match.group("model"),
                                match.group("dataset"),
                                match.group("variant"),
                                match.group("init_layers_X"),
                                match.group("structure_info"),
                                match.group("hidden"),
                                match.group("lr"),
                                match.group("weight_decay"),
                                match.group("dropout"),
                                match.group("hops"),
                                match.group("resnet"),
                                match.group("layer_norm"),
                                match.group("att_hopwise_distinct"),
                                match.group("fuse_hop"),
                                match.group("layers"),
                                match.group("test_mean"),
                                match.group("test_std"),
                                match.group("runtime_avg"),
                                match.group("epoch_avg"),
                                match.group("normalization"),
                                match.group("online_cs"),
                                match.group("lambda_pen"),
                                match.group("lambda_2hop"),
                                match.group("threshold"),
                                match.group("svd_rank"),
                                match.group("top"),
                                match.group("comm_size"),
                                match.group("cs_time"),
                                match.group("cs_nmi"),
                                match.group("cs_nmi_std"),
                                match.group("cs_jaccard"),
                                match.group("cs_jaccard_std"),
                                match.group("cs_f1"),
                                match.group("cs_f1_std")
                            ])
                        else:
                            print(f"Skipping unmatched line: {line.strip()}")

    print(f"\nAll logs from {start_date_str} to {end_date_str} have been saved to {master_csv_path}")

# 示例用法
process_logs_for_date_range("2025_03_28", "2025_03_31")
=======
# import re
# import csv
# import os
# from glob import glob

# # Folder path for logs
# log_folder_path = "logs/"

# # Updated regular expression to extract all parameters from the log line.
# # This regex assumes log lines of the form:
# #
# # 2025-02-17 10:29:55: Run 9 Summary - model=acmsmn, dataset=texas, variant=1, init_layers_X=2, structure_info=2, Hidden=1024, lr=0.0100, weight_decay=0.001000, dropout=0.2000, hops=9, resnet=0, layer_norm=0, att_hopwise_distinct=0, fuse_hop=mlp, layers=1, Test_Mean=0.9197, Test_Std=0.0248, runtime_average=17.02s, epoch_average=171.94ms, normalization=local_w, online_cs=sub_cs, lambda_pen=1.0, lambda_2hop=0.5, threshold=0.9, comm_size=20, CommunitySearchTime=0.027379s, NMI=0.0000 (std=0.0000), Jaccard=0.9367 (std=0.0283), F1=0.9671 (std=0.0141)
# #
# log_pattern = re.compile(
#     r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Run (?P<run>\d+) Summary - "
#     r"model=(?P<model>\w+), dataset=(?P<dataset>\w+), variant=(?P<variant>[\d.]+), "
#     r"init_layers_X=(?P<init_layers_X>\d+), structure_info=(?P<structure_info>\d+), "
#     r"Hidden=(?P<hidden>\d+), lr=(?P<lr>[\d.]+), weight_decay=(?P<weight_decay>[\d.]+), "
#     r"dropout=(?P<dropout>[\d.]+), hops=(?P<hops>\d+), resnet=(?P<resnet>\d+), "
#     r"layer_norm=(?P<layer_norm>\d+), att_hopwise_distinct=(?P<att_hopwise_distinct>\d+), "
#     r"fuse_hop=(?P<fuse_hop>\w+), layers=(?P<layers>\d+), "
#     r"Test_Mean=(?P<test_mean>[\d.]+), Test_Std=(?P<test_std>[\d.]+), "
#     r"runtime_average=(?P<runtime_avg>[\d.]+)s, epoch_average=(?P<epoch_avg>[\d.]+)ms, "
#     r"normalization=(?P<normalization>\w+), online_cs=(?P<online_cs>\w+), "
#     r"lambda_pen=(?P<lambda_pen>[\d.]+), lambda_2hop=(?P<lambda_2hop>[\d.]+), "
#     r"threshold=(?P<threshold>[\d.]+), comm_size=(?P<comm_size>\d+), "
#     r"CommunitySearchTime=(?P<cs_time>[\d.]+)s, "
#     r"NMI=(?P<cs_nmi>[\d.]+) \(std=(?P<cs_nmi_std>[\d.]+)\), "
#     r"Jaccard=(?P<cs_jaccard>[\d.]+) \(std=(?P<cs_jaccard_std>[\d.]+)\), "
#     r"F1=(?P<cs_f1>[\d.]+) \(std=(?P<cs_f1_std>[\d.]+)\)"
# )

# def process_logs_for_date(date):
#     # Format the CSV file path to include the date
#     csv_file_path = f"results_csv/acm_torch_logs_{date}.csv"
#     os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

#     with open(csv_file_path, mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         # Write the header row with all parameters
#         writer.writerow([
#             "Timestamp", "Run", "Model", "Dataset", "Variant", "Init Layers X",
#             "Structure Info", "Hidden", "lr", "weight_decay", "dropout", "hops",
#             "resnet", "layer_norm", "att_hopwise_distinct", "fuse_hop", "layers",
#             "Test Mean", "Test Std", "runtime_average", "epoch_average", "normalization",
#             "online_cs", "lambda_pen", "lambda_2hop", "threshold", "comm_size",
#             "CommunitySearchTime", "NMI", "NMI_std", "Jaccard", "Jaccard_std", "F1", "F1_std"
#         ])

#         # Iterate through all log files in the logs folder that contain the given date in the filename
#         for log_file_path in glob(os.path.join(log_folder_path, f"*{date}*.log")):
#             print(f"Processing file: {log_file_path}")
#             with open(log_file_path, 'r') as log_file:
#                 for line in log_file:
#                     match = log_pattern.match(line)
#                     if match:
#                         writer.writerow([
#                             match.group("timestamp"),
#                             match.group("run"),
#                             match.group("model"),
#                             match.group("dataset"),
#                             match.group("variant"),
#                             match.group("init_layers_X"),
#                             match.group("structure_info"),
#                             match.group("hidden"),
#                             match.group("lr"),
#                             match.group("weight_decay"),
#                             match.group("dropout"),
#                             match.group("hops"),
#                             match.group("resnet"),
#                             match.group("layer_norm"),
#                             match.group("att_hopwise_distinct"),
#                             match.group("fuse_hop"),
#                             match.group("layers"),
#                             match.group("test_mean"),
#                             match.group("test_std"),
#                             match.group("runtime_avg"),
#                             match.group("epoch_avg"),
#                             match.group("normalization"),
#                             match.group("online_cs"),
#                             match.group("lambda_pen"),
#                             match.group("lambda_2hop"),
#                             match.group("threshold"),
#                             match.group("comm_size"),
#                             match.group("cs_time"),
#                             match.group("cs_nmi"),
#                             match.group("cs_nmi_std"),
#                             match.group("cs_jaccard"),
#                             match.group("cs_jaccard_std"),
#                             match.group("cs_f1"),
#                             match.group("cs_f1_std")
#                         ])
#                         print(f"Matched and wrote: {match.groups()}")
#                     else:
#                         print(f"No match for line: {line}")
#     print(f"Logs have been successfully saved to {csv_file_path}.")

# # Example usage: Process logs for a specific date (adjust date as needed)
# process_logs_for_date("2025_02_17")

import re
import csv
import os
from glob import glob
from datetime import datetime, timedelta

# Folder path for logs
log_folder_path = "logs/"

log_pattern = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): Run (?P<run>\d+) Summary - "
    r"model=(?P<model>\w+), dataset=(?P<dataset>\w+), variant=(?P<variant>[\d.]+), "
    r"init_layers_X=(?P<init_layers_X>\d+), structure_info=(?P<structure_info>\d+), "
    r"Hidden=(?P<hidden>\d+), lr=(?P<lr>[\d.]+), weight_decay=(?P<weight_decay>[\d.]+), "
    r"dropout=(?P<dropout>[\d.]+), hops=(?P<hops>\d+), resnet=(?P<resnet>\d+), "
    r"layer_norm=(?P<layer_norm>\d+), att_hopwise_distinct=(?P<att_hopwise_distinct>\d+), "
    r"fuse_hop=(?P<fuse_hop>\w+), layers=(?P<layers>\d+), "
    r"Test_Mean=(?P<test_mean>[\d.]+), Test_Std=(?P<test_std>[\d.]+), "
    r"runtime_average=(?P<runtime_avg>[\d.]+)s, epoch_average=(?P<epoch_avg>[\d.]+)ms, "
    r"normalization=(?P<normalization>\w+), online_cs=(?P<online_cs>\w+), "
    r"lambda_pen=(?P<lambda_pen>[\d.]+), lambda_2hop=(?P<lambda_2hop>[\d.]+), "
    r"threshold=(?P<threshold>[\d.]+), comm_size=(?P<comm_size>\d+), "
    r"CommunitySearchTime=(?P<cs_time>[\d.]+)s, "
    r"NMI=(?P<cs_nmi>[\d.]+) \(std=(?P<cs_nmi_std>[\d.]+)\), "
    r"Jaccard=(?P<cs_jaccard>[\d.]+) \(std=(?P<cs_jaccard_std>[\d.]+)\), "
    r"F1=(?P<cs_f1>[\d.]+) \(std=(?P<cs_f1_std>[\d.]+)\)"
)

def parse_date(date_str, input_format="%Y_%m_d"):
    """将字符串日期转换为datetime对象"""
    return datetime.strptime(date_str, input_format)

def date_range(start_date, end_date):
    """生成日期范围内所有日期的生成器"""
    delta = end_date - start_date
    for i in range(delta.days + 1):
        yield start_date + timedelta(days=i)

def process_logs_for_date_range(start_date_str, end_date_str):
    """
    处理指定日期范围内的所有日志文件
    :param start_date_str: 开始日期字符串 (格式: "YYYY_MM_DD")
    :param end_date_str: 结束日期字符串 (格式: "YYYY_MM_DD")
    """
    # 转换输入日期
    start_date = parse_date(start_date_str, "%Y_%m_%d")
    end_date = parse_date(end_date_str, "%Y_%m_%d")
    
    # 创建主CSV文件
    master_csv_path = f"results_csv/master_logs_{start_date_str}_to_{end_date_str}.csv"
    os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)
    
    with open(master_csv_path, mode='w', newline='') as master_csv:
        writer = csv.writer(master_csv)
        # 只写一次表头
        writer.writerow([
            "Timestamp", "Run", "Model", "Dataset", "Variant", "Init Layers X",
            "Structure Info", "Hidden", "lr", "weight_decay", "dropout", "hops",
            "resnet", "layer_norm", "att_hopwise_distinct", "fuse_hop", "layers",
            "Test Mean", "Test Std", "runtime_average", "epoch_average", "normalization",
            "online_cs", "lambda_pen", "lambda_2hop", "threshold", "svd_rank", "top", "comm_size",
            "CommunitySearchTime", "NMI", "NMI_std", "Jaccard", "Jaccard_std", "F1", "F1_std"
        ])

        # 遍历日期范围内的每一天
        for single_date in date_range(start_date, end_date):
            date_str = single_date.strftime("%Y_%m_%d")
            print(f"\nProcessing date: {date_str}")
            
            # 查找匹配的日志文件
            log_files = glob(os.path.join(log_folder_path, f"*{date_str}*.log"))
            
            if not log_files:
                print(f"No log files found for date {date_str}")
                continue
                
            # 处理每个日志文件
            for log_file_path in log_files:
                print(f"Processing file: {log_file_path}")
                with open(log_file_path, 'r') as log_file:
                    for line in log_file:
                        match = log_pattern.match(line)
                        if match:
                            writer.writerow([
                                match.group("timestamp"),
                                match.group("run"),
                                match.group("model"),
                                match.group("dataset"),
                                match.group("variant"),
                                match.group("init_layers_X"),
                                match.group("structure_info"),
                                match.group("hidden"),
                                match.group("lr"),
                                match.group("weight_decay"),
                                match.group("dropout"),
                                match.group("hops"),
                                match.group("resnet"),
                                match.group("layer_norm"),
                                match.group("att_hopwise_distinct"),
                                match.group("fuse_hop"),
                                match.group("layers"),
                                match.group("test_mean"),
                                match.group("test_std"),
                                match.group("runtime_avg"),
                                match.group("epoch_avg"),
                                match.group("normalization"),
                                match.group("online_cs"),
                                match.group("lambda_pen"),
                                match.group("lambda_2hop"),
                                match.group("threshold"),
                                match.group("svd_rank"),
                                match.group("top"),
                                match.group("comm_size"),
                                match.group("cs_time"),
                                match.group("cs_nmi"),
                                match.group("cs_nmi_std"),
                                match.group("cs_jaccard"),
                                match.group("cs_jaccard_std"),
                                match.group("cs_f1"),
                                match.group("cs_f1_std")
                            ])
                        else:
                            print(f"Skipping unmatched line: {line.strip()}")

    print(f"\nAll logs from {start_date_str} to {end_date_str} have been saved to {master_csv_path}")

# 示例用法
process_logs_for_date_range("2025_03_28", "2025_03_31")
>>>>>>> origin/main
