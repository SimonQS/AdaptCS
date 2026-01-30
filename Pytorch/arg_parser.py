import argparse
import torch


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument(
        "--device", type=int, default=7, help="CUDA device id to use if available."
    )
    parser.add_argument(
        "--param_tunning",
        action="store_true",
        default=True,
        help="Parameter fine-tunning mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default= 100, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num_splits", type=int, help="number of training/val/test splits ", default=10
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "adaptcs"],
        help="name of the model",
        default="adaptcs",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="optimizer for large datasets (Adam, AdamW)",
        default="AdamW",
    )
    parser.add_argument(
        "--early_stopping", type=float, default=200, help="early stopping used in GPRGNN"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument("--hidden", type=int, default=512, help="Number of hidden units.")
    parser.add_argument("--hops", type=int, default=5, help="Number of hops we use, k= 1,2")
    parser.add_argument("--svd_rank", type=int, default=100, help="Number of rank for svd")
    parser.add_argument("--svd_power_iters", type=int, default=10, help="Number of iteration for randomnised svd")
    parser.add_argument(
        "--layers", type=int, default=2, help="Number of hidden layers, i.e. network depth"
    )
    parser.add_argument(
        "--link_init_layers_X", type=int, default=2, help="Number of initial layer"
    )
    parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="cornell")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
    )
    parser.add_argument(
        "--fixed_splits",
        type=float,
        default=0,
        help="0 for random splits in GPRGNN, 1 for fixed splits in GeomGCN",
    )
    parser.add_argument(
        "--variant", 
        type=float, 
        default=1, 
        help="Indicate ACM, GCNII variant models. 0, 1"
    )
    parser.add_argument(
        "--normalization", 
        type=str, 
        default='global_w', 
        help="Apply node-wise local attention as normalization.",
        choices=['local_w', 
                 'global_w', 
                 'softmax', 
                 'row_sum',
                 'none',
                 'sparse_row_sum',
                 'sparse_row_sum_gain',
                 'global_gain',
                 'global_scale',
                 'log',]
    )
    parser.add_argument(
        "--resnet", 
        type=float, 
        default=0, 
        help="Apply resnet to add self-features, Xx."
    )
    parser.add_argument(
        "--layer_norm", 
        type=float, 
        default=1, 
        help="Apply layer normalization to attention."
    )
    parser.add_argument(
        "--att_hopwise_distinct", 
        type=float, 
        default=0, 
        help="1 for each hop will have a distinct attention weight matrx, 0 share between hops"
    )
    parser.add_argument(
        "--fuse_hop",
        type=str,
        choices=[
            "self",
            "cat",
            "qkv",
            "mlp",
            "bank",
        ],
        help="How to fuse the multi-hop channels",
        default="bank",
    )
    parser.add_argument(
        "--online_cs",
        type=str,
        choices=[
            "signed_cs",
            "sub_cs",
            "sub_topk",
            "bfs_teleport",
            "adaptive_cs",
            "bfs",
        ],
        help="Online serch algoirithms to use",
        default="adaptive_cs",
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=[
            "distinct_hop",
            "distinct_hop_svds_low",
            "distinct_hop_svds_rand",
        ],
        help="which approach to use for the message passing",
        default="distinct_hop",
    )
    parser.add_argument(
        "--lambda_pen",
        type=float,
        default=0.5,
        help="Penalty weight (only relevant if --online_cs=signed_cs)",
    )
    parser.add_argument(
        "--lambda_2hop",
        type=float,
        default=0.5,
        help="2-hop bonus weight (only relevant if --online_cs=signed_cs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="threshold for negative edge (only relevant if --online_cs= (signed_cs, bfs_teleport))",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=2,
        help="top factor",
    )
    parser.add_argument(
        "--structure_info",
        type=int,
        default=1,
        help="1 for using structure information in acmgcnp, 0 for not, 2 for attention2",
    )
    parser.add_argument(
        "--comm_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch_csv",
        type=str,
        default=None,
        help="save the batch csv file"
    )
    parser.add_argument(
        "--masking",
        type=str,
        choices=["hard", "adaptive"],
        default="adaptive",
        help="Masking strategy for distinct_hop: 'hard' masks all previously seen edges, 'adaptive' uses ReLU(A^k - A^(k-1))"
    )
    args = parser.parse_args()


    DATASET_DEFAULTS = {
    'cornell':   dict(comm_size=30),
    'wisconsin': dict(comm_size=30),
    'texas':     dict(comm_size=30),
    'cora':      dict(comm_size=150),
    'citeseer':  dict(comm_size=150),
    'pubmed':    dict(comm_size=150),
    'film':      dict(comm_size=150),
    'chameleon': dict(comm_size=150),
    'squirrel':  dict(comm_size=150),
    'reddit':    dict(comm_size=1000)}
    
    if args.dataset_name == 'reddit':
        args.hidden = 128
        args.hops = 3
        args.svd_rank = 50
    
    if args.model != 'adaptcs': 
        args.hidden = 128
        args.hop = 3
    

    for key, val in DATASET_DEFAULTS.get(args.dataset_name, {}).items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args

