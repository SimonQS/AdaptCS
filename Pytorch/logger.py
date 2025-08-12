# import sys
# sys.path.insert(0, "..")
# from BaseLogger import BaseLogger

# class ACMPythorchLogger(BaseLogger):
#     def __init__(self, csv_path=None):
#         """
#         :param csv_path: Optional path to CSV for batch run outputs.
#         """
#         super().__init__("acm-torch-logger")
#         self.csv_path = csv_path
#         if self.csv_path:
#             import os, csv
#             os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
#             is_new = not os.path.exists(self.csv_path)
#             self.csv_file = open(self.csv_path, "a", newline="")
#             self.csv_writer = csv.writer(self.csv_file)
#             if is_new:
#                 self.csv_writer.writerow([
#                     "model","dataset_name","variant","structure_info","init_layers_X",
#                     "hidden","layers","hops","fixed_splits","resnet","layer_norm",
#                     "att_hopwise_distinct","fuse_hop","normalization","online_cs",
#                     "lambda_pen","lambda_2hop","threshold","comm_size","approach",
#                     "split","lr","weight_decay","dropout",
#                     "test_mean","test_std","runtime_average","epoch_average",
#                     "cs_time","cs_nmi","cs_nmi_std","cs_jaccard","cs_jaccard_std","cs_f1","cs_f1_std"
#                 ])
#                 self.csv_file.flush()

#     def log_best_result(self, model_info, best_result_info):
#         """
#         Logs final/best result. We rely on 'model_info' to store all relevant args,
#         and 'best_result_info' to store final performance metrics.
#         """
#         msg = (
#             f"Best Result - "
#             f"model={model_info['model']}, "
#             f"dataset={model_info['dataset_name']}, "
#             f"variant={model_info['variant']}, "
#             f"structure_info={model_info['structure_info']}, "
#             f"fixed_splits={model_info['fixed_splits']}, "
#             f"Hidden={model_info['hidden']}, "
#             f"layers={model_info['layers']}, "
#             f"hops={model_info['hops']}, "
#             f"resnet={model_info['resnet']}, "
#             f"layer_norm={model_info['layer_norm']}, "
#             f"att_hopwise_distinct={model_info['att_hopwise_distinct']}, "
#             f"fuse_hop={model_info.get('fuse_hop')}, "
#             f"normalization={model_info['normalization']}, "
#             f"online_cs={model_info['online_cs']}, "
#             f"lambda_pen={model_info['lambda_pen']}, "
#             f"lambda_2hop={model_info['lambda_2hop']}, "
#             f"threshold={model_info['threshold']}, "
#             f"comm_size={model_info['comm_size']}, "
#             f"approach={model_info.get('approach')}, "
#             f"optimizer={model_info.get('optimizer')}, "
#             f"early_stopping={model_info.get('early_stopping')}, "
#             f"lr={best_result_info['lr']:.4f}, "
#             f"weight_decay={best_result_info['weight_decay']:.6f}, "
#             f"dropout={best_result_info['dropout']:.4f}, "
#             f"Test Mean={best_result_info['test_result']:.4f}, "
#             f"Test Std={best_result_info['test_std']:.4f}, "
#             f"epoch avg/runtime avg time="
#             f"{best_result_info['epoch_average']:.2f}ms/{best_result_info['runtime_average']:.2f}s"
#         )
#         self.logger.info(msg)

#     def log_param_tune(
#         self,
#         model_info,
#         curr_split,
#         curr_dropout,
#         curr_weight_decay,
#         curr_lr,
#         curr_res,
#         curr_loss
#     ):
#         """
#         Logs info during parameter tuning.
#         We show relevant hyperparams used in the run.
#         """
#         msg = (
#             f"Optimization - "
#             f"model={model_info['model']}, "
#             f"dataset={model_info['dataset_name']}, "
#             f"variant={model_info['variant']}, "
#             f"structure_info={model_info['structure_info']}, "
#             f"fixed_splits={model_info['fixed_splits']}, "
#             f"hidden={model_info['hidden']}, "
#             f"layers={model_info['layers']}, "
#             f"hops={model_info['hops']}, "
#             f"resnet={model_info['resnet']}, "
#             f"layer_norm={model_info['layer_norm']}, "
#             f"att_hopwise_distinct={model_info['att_hopwise_distinct']}, "
#             f"normalization={model_info['normalization']}, "
#             f"online_cs={model_info['online_cs']}, "
#             f"lambda_pen={model_info['lambda_pen']}, "
#             f"lambda_2hop={model_info['lambda_2hop']}, "
#             f"threshold={model_info['threshold']}, "
#             f"comm_size={model_info['comm_size']}, "
#             f"approach={model_info.get('approach')}, "
#             f"split={curr_split}, "
#             f"lr={curr_lr:.5f}, "
#             f"weight_decay={curr_weight_decay:.5f}, "
#             f"dropout={curr_dropout:.4f}, "
#             f"Best_Test_Result={curr_res:.4f}, "
#             f"Training_Loss={curr_loss:.4f}"
#         )
#         self.logger.info(msg)

#     def log_run(self, model_info, run_info):
#         """
#         Logs summary info for a single run or split.
#         """
#         # CSV write if requested
#         if self.csv_path:
#             row = [
#                 model_info['model'],
#                 model_info['dataset_name'],
#                 model_info['variant'],
#                 model_info['structure_info'],
#                 model_info.get('init_layers_X'),
#                 model_info['hidden'],
#                 model_info['layers'],
#                 model_info['hops'],
#                 model_info['fixed_splits'],
#                 model_info['resnet'],
#                 model_info['layer_norm'],
#                 model_info['att_hopwise_distinct'],
#                 model_info.get('fuse_hop'),
#                 model_info['normalization'],
#                 model_info['online_cs'],
#                 model_info['lambda_pen'],
#                 model_info['lambda_2hop'],
#                 model_info['threshold'],
#                 model_info['comm_size'],
#                 model_info.get('approach'),
#                 run_info['split'],
#                 run_info['lr'],
#                 run_info['weight_decay'],
#                 run_info['dropout'],
#                 run_info['result'],
#                 run_info['std'],
#                 run_info['runtime_average'],
#                 run_info['epoch_average'],
#                 run_info.get('cs_time',''),
#                 run_info.get('cs_nmi',''),
#                 run_info.get('cs_nmi_std',''),
#                 run_info.get('cs_jaccard',''),
#                 run_info.get('cs_jaccard_std',''),
#                 run_info.get('cs_f1',''),
#                 run_info.get('cs_f1_std','')
#             ]
#             self.csv_writer.writerow(row)
#             self.csv_file.flush()

#         # Standard log
#         msg = (
#             f"Run {run_info['split']} Summary - "
#             f"model={model_info['model']}, "
#             f"dataset={model_info['dataset_name']}, "
#             f"variant={model_info['variant']}, "
#             f"structure_info={model_info['structure_info']}, "
#             f"fixed_splits={model_info['fixed_splits']}, "
#             f"Hidden={model_info['hidden']}, "
#             f"layers={model_info['layers']}, "
#             f"hops={model_info['hops']}, "
#             f"resnet={model_info['resnet']}, "
#             f"layer_norm={model_info['layer_norm']}, "
#             f"att_hopwise_distinct={model_info['att_hopwise_distinct']}, "
#             f"fuse_hop={model_info.get('fuse_hop')}, "
#             f"normalization={model_info['normalization']}, "
#             f"online_cs={model_info['online_cs']}, "
#             f"lambda_pen={model_info['lambda_pen']}, "
#             f"lambda_2hop={model_info['lambda_2hop']}, "
#             f"threshold={model_info['threshold']}, "
#             f"comm_size={model_info['comm_size']}, "
#             f"approach={model_info.get('approach')}, "
#             f"lr={run_info['lr']:.4f}, "
#             f"weight_decay={run_info['weight_decay']:.6f}, "
#             f"dropout={run_info['dropout']:.4f}, "
#             f"Test_Mean={run_info['result']:.4f}, "
#             f"Test_Std={run_info['std']:.4f}, "
#             f"runtime_average={run_info['runtime_average']:.2f}s, "
#             f"epoch_average={run_info['epoch_average']:.2f}ms"
#         )
#         if "cs_time" in run_info:
#             msg += (
#                 f", CommunitySearchTime={run_info['cs_time']:.6f}s, "
#                 f"NMI={run_info['cs_nmi']:.4f} (std={run_info['cs_nmi_std']:.4f}), "
#                 f"Jaccard={run_info['cs_jaccard']:.4f} (std={run_info['cs_jaccard_std']:.4f}), "
#                 f"F1={run_info['cs_f1']:.4f} (std={run_info['cs_f1_std']:.4f})"
#             )
#         self.logger.info(msg)

import sys, os, csv
sys.path.insert(0, "..")
from BaseLogger import BaseLogger


def _fmt(value, spec: str = "", na: str = "NA"):
    """
    Safe formatter.
    - 如果 value 为 None 或不存在 → 返回占位符 na
    - 否则按 spec（如 '.4f'）格式化；若 spec 为空直接 str(value)
    """
    if value is None:
        return na
    return f"{value:{spec}}" if spec else str(value)


class ACMPythorchLogger(BaseLogger):
    def __init__(self, csv_path: str | None = None):
        super().__init__("acm-torch-logger")

        self.csv_path = csv_path
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            is_new = not os.path.exists(self.csv_path)
            self.csv_file = open(self.csv_path, "a", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            if is_new:
                self.csv_writer.writerow([
                    "model", "dataset_name", "variant", "structure_info", "init_layers_X",
                    "hidden", "layers", "hops", "fixed_splits", "resnet", "layer_norm",
                    "att_hopwise_distinct", "fuse_hop", "normalization", "online_cs",
                    "lambda_pen", "lambda_2hop", "threshold", "svd_rank", "top", 
                    "comm_size", "approach", "split", "lr", "weight_decay", "dropout",
                    "test_mean", "test_std", "runtime_average", "epoch_average",
                    "cs_time", "cs_nmi", "cs_nmi_std", "cs_jaccard",
                    "cs_jaccard_std", "cs_f1", "cs_f1_std"
                ])
                self.csv_file.flush()

    # ------------------------------------------------------------------ #
    # Best result
    # ------------------------------------------------------------------ #
    def log_best_result(self, model_info: dict, best_result_info: dict):
        msg = (
            "Best Result - "
            f"model={_fmt(model_info.get('model'))}, "
            f"dataset={_fmt(model_info.get('dataset_name'))}, "
            f"variant={_fmt(model_info.get('variant'))}, "
            f"structure_info={_fmt(model_info.get('structure_info'))}, "
            f"fixed_splits={_fmt(model_info.get('fixed_splits'))}, "
            f"Hidden={_fmt(model_info.get('hidden'))}, "
            f"layers={_fmt(model_info.get('layers'))}, "
            f"hops={_fmt(model_info.get('hops'))}, "
            f"resnet={_fmt(model_info.get('resnet'))}, "
            f"layer_norm={_fmt(model_info.get('layer_norm'))}, "
            f"att_hopwise_distinct={_fmt(model_info.get('att_hopwise_distinct'))}, "
            f"fuse_hop={_fmt(model_info.get('fuse_hop'))}, "
            f"normalization={_fmt(model_info.get('normalization'))}, "
            f"online_cs={_fmt(model_info.get('online_cs'))}, "
            f"lambda_pen={_fmt(model_info.get('lambda_pen'))}, "
            f"lambda_2hop={_fmt(model_info.get('lambda_2hop'))}, "
            f"threshold={_fmt(model_info.get('threshold'))}, "
            f"svd_rank={_fmt(model_info.get('svd_rank'))}, "
            f"top={_fmt(model_info.get('top'))}, "
            f"comm_size={_fmt(model_info.get('comm_size'))}, "
            f"approach={_fmt(model_info.get('approach'))}, "
            f"optimizer={_fmt(model_info.get('optimizer'))}, "
            f"early_stopping={_fmt(model_info.get('early_stopping'))}, "
            f"lr={_fmt(best_result_info.get('lr'), '.4f')}, "
            f"weight_decay={_fmt(best_result_info.get('weight_decay'), '.6f')}, "
            f"dropout={_fmt(best_result_info.get('dropout'), '.4f')}, "
            f"Test Mean={_fmt(best_result_info.get('test_result'), '.4f')}, "
            f"Test Std={_fmt(best_result_info.get('test_std'), '.4f')}, "
            f"epoch avg/runtime avg time="
            f"{_fmt(best_result_info.get('epoch_average'), '.2f')}ms/"
            f"{_fmt(best_result_info.get('runtime_average'), '.2f')}s"
        )
        self.logger.info(msg)

    # ------------------------------------------------------------------ #
    # During parameter tuning
    # ------------------------------------------------------------------ #
    def log_param_tune(
        self,
        model_info: dict,
        curr_split,
        curr_dropout,
        curr_weight_decay,
        curr_lr,
        curr_res,
        curr_loss,
    ):
        msg = (
            "Optimization - "
            f"model={_fmt(model_info.get('model'))}, "
            f"dataset={_fmt(model_info.get('dataset_name'))}, "
            f"variant={_fmt(model_info.get('variant'))}, "
            f"structure_info={_fmt(model_info.get('structure_info'))}, "
            f"fixed_splits={_fmt(model_info.get('fixed_splits'))}, "
            f"hidden={_fmt(model_info.get('hidden'))}, "
            f"layers={_fmt(model_info.get('layers'))}, "
            f"hops={_fmt(model_info.get('hops'))}, "
            f"resnet={_fmt(model_info.get('resnet'))}, "
            f"layer_norm={_fmt(model_info.get('layer_norm'))}, "
            f"att_hopwise_distinct={_fmt(model_info.get('att_hopwise_distinct'))}, "
            f"normalization={_fmt(model_info.get('normalization'))}, "
            f"online_cs={_fmt(model_info.get('online_cs'))}, "
            f"lambda_pen={_fmt(model_info.get('lambda_pen'))}, "
            f"lambda_2hop={_fmt(model_info.get('lambda_2hop'))}, "
            f"threshold={_fmt(model_info.get('threshold'))}, "
            f"svd_rank={_fmt(model_info.get('svd_rank'))}, "
            f"top={_fmt(model_info.get('top'))}, "
            f"comm_size={_fmt(model_info.get('comm_size'))}, "
            f"approach={_fmt(model_info.get('approach'))}, "
            f"split={_fmt(curr_split)}, "
            f"lr={_fmt(curr_lr, '.5f')}, "
            f"weight_decay={_fmt(curr_weight_decay, '.5f')}, "
            f"dropout={_fmt(curr_dropout, '.4f')}, "
            f"Best_Test_Result={_fmt(curr_res, '.4f')}, "
            f"Training_Loss={_fmt(curr_loss, '.4f')}"
        )
        self.logger.info(msg)

    # ------------------------------------------------------------------ #
    # Per-run summary
    # ------------------------------------------------------------------ #
    def log_run(self, model_info: dict, run_info: dict):
        # ---------- CSV ---------- #
        if self.csv_path:
            self.csv_writer.writerow([
                _fmt(model_info.get('model'), na=''),
                _fmt(model_info.get('dataset_name'), na=''),
                _fmt(model_info.get('variant'), na=''),
                _fmt(model_info.get('structure_info'), na=''),
                _fmt(model_info.get('init_layers_X'), na=''),
                _fmt(model_info.get('hidden'), na=''),
                _fmt(model_info.get('layers'), na=''),
                _fmt(model_info.get('hops'), na=''),
                _fmt(model_info.get('fixed_splits'), na=''),
                _fmt(model_info.get('resnet'), na=''),
                _fmt(model_info.get('layer_norm'), na=''),
                _fmt(model_info.get('att_hopwise_distinct'), na=''),
                _fmt(model_info.get('fuse_hop'), na=''),
                _fmt(model_info.get('normalization'), na=''),
                _fmt(model_info.get('online_cs'), na=''),
                _fmt(model_info.get('lambda_pen'), na=''),
                _fmt(model_info.get('lambda_2hop'), na=''),
                _fmt(model_info.get('threshold'), na=''),
                _fmt(model_info.get('svd_rank'), na=''),
                _fmt(model_info.get('top'), na=''),
                _fmt(model_info.get('comm_size'), na=''),
                _fmt(model_info.get('approach'), na=''),
                _fmt(run_info.get('split'), na=''),
                _fmt(run_info.get('lr'), na=''),
                _fmt(run_info.get('weight_decay'), na=''),
                _fmt(run_info.get('dropout'), na=''),
                _fmt(run_info.get('result'), na=''),
                _fmt(run_info.get('std'), na=''),
                _fmt(run_info.get('runtime_average'), na=''),
                _fmt(run_info.get('epoch_average'), na=''),
                _fmt(run_info.get('cs_time'), na=''),
                _fmt(run_info.get('cs_nmi'), na=''),
                _fmt(run_info.get('cs_nmi_std'), na=''),
                _fmt(run_info.get('cs_jaccard'), na=''),
                _fmt(run_info.get('cs_jaccard_std'), na=''),
                _fmt(run_info.get('cs_f1'), na=''),
                _fmt(run_info.get('cs_f1_std'), na='')
            ])
            self.csv_file.flush()

        # ---------- Human-readable log ---------- #
        msg = (
            f"Run {_fmt(run_info.get('split'))} Summary - "
            f"model={_fmt(model_info.get('model'))}, "
            f"dataset={_fmt(model_info.get('dataset_name'))}, "
            f"variant={_fmt(model_info.get('variant'))}, "
            f"structure_info={_fmt(model_info.get('structure_info'))}, "
            f"fixed_splits={_fmt(model_info.get('fixed_splits'))}, "
            f"Hidden={_fmt(model_info.get('hidden'))}, "
            f"layers={_fmt(model_info.get('layers'))}, "
            f"hops={_fmt(model_info.get('hops'))}, "
            f"resnet={_fmt(model_info.get('resnet'))}, "
            f"layer_norm={_fmt(model_info.get('layer_norm'))}, "
            f"att_hopwise_distinct={_fmt(model_info.get('att_hopwise_distinct'))}, "
            f"fuse_hop={_fmt(model_info.get('fuse_hop'))}, "
            f"normalization={_fmt(model_info.get('normalization'))}, "
            f"online_cs={_fmt(model_info.get('online_cs'))}, "
            f"lambda_pen={_fmt(model_info.get('lambda_pen'))}, "
            f"lambda_2hop={_fmt(model_info.get('lambda_2hop'))}, "
            f"threshold={_fmt(model_info.get('threshold'))}, "
            f"svd_rank={_fmt(model_info.get('svd_rank'))}, "
            f"top={_fmt(model_info.get('top'))}, "
            f"comm_size={_fmt(model_info.get('comm_size'))}, "
            f"approach={_fmt(model_info.get('approach'))}, "
            f"lr={_fmt(run_info.get('lr'), '.4f')}, "
            f"weight_decay={_fmt(run_info.get('weight_decay'), '.6f')}, "
            f"dropout={_fmt(run_info.get('dropout'), '.4f')}, "
            f"Test_Mean={_fmt(run_info.get('result'), '.4f')}, "
            f"Test_Std={_fmt(run_info.get('std'), '.4f')}, "
            f"runtime_average={_fmt(run_info.get('runtime_average'), '.2f')}s, "
            f"epoch_average={_fmt(run_info.get('epoch_average'), '.2f')}ms"
        )

        # 追加社区搜索度量（如果存在）
        if run_info.get("cs_time") is not None:
            msg += (
                f", CommunitySearchTime={_fmt(run_info['cs_time'], '.6f')}s, "
                f"NMI={_fmt(run_info['cs_nmi'], '.4f')} "
                f"(std={_fmt(run_info['cs_nmi_std'], '.4f')}), "
                f"Jaccard={_fmt(run_info['cs_jaccard'], '.4f')} "
                f"(std={_fmt(run_info['cs_jaccard_std'], '.4f')}), "
                f"F1={_fmt(run_info['cs_f1'], '.4f')} "
                f"(std={_fmt(run_info['cs_f1_std'], '.4f')})"
            )

        self.logger.info(msg)
