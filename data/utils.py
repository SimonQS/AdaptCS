import os
from os import path
import pickle as pkl
import sys
import heapq
from google_drive_downloader import GoogleDriveDownloader as gdd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy
import scipy.io
import scipy.sparse
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize as sk_normalize
from time import perf_counter
import torch
import torch.nn.functional as F
from arg_parser import arg_parser
from collections import deque
from typing import List

if torch.cuda.is_available():
    import scipy.io
    from torch_geometric.utils import add_self_loops, to_undirected

    from torch_sparse import SparseTensor
args = arg_parser()


DATA_PATH = path.dirname(path.abspath(__file__)) + "/data/"
SPLITS_DRIVE_URL = {
    "snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N",
    "pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_",
}

device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)


class NCDataset(object):
    def __init__(self, name, root=f"{DATA_PATH}"):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def data_split(idx, dataset_name):
    splits_file_path = "splits/" + dataset_name + "_split_0.6_0.2_" + str(idx) + ".npz"
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file["train_mask"]
        val_mask = splits_file["val_mask"]
        test_mask = splits_file["test_mask"]
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    return train_mask, val_mask, test_mask


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


@torch.no_grad()
def evaluate(output, labels, split_idx, eval_func):
    acc = eval_func(labels[split_idx], output[split_idx])
    return acc


def eval_acc(y_true, y_pred):
    if y_true.dim() > 1:
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return sum(acc_list) / len(acc_list)
    else:
        preds = y_pred.max(1)[1].type_as(y_true)
        correct = preds.eq(y_true).double()
        correct = correct.sum()
        return correct / len(y_true)


def eval_rocauc(y_true, y_pred):
    """
    adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py
    """
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return sum(rocauc_list) / len(rocauc_list)


def even_quantile_labels(vals, nclasses, verbose=True):
    """
    partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high


def gen_normalized_adjs(dataset):
    """
    returns the normalized adjacency matrix
    """
    dataset.graph["edge_index"] = add_self_loops(dataset.graph["edge_index"])[0]
    row, col = dataset.graph["edge_index"]
    N = dataset.graph["num_nodes"]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float("inf")] = 0
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def load_deezer_dataset():
    filename = "deezer-europe"
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f"{DATA_PATH}deezer-europe.mat")
    A, label, features = deezer["A"], deezer["label"], deezer["features"]
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {
        "edge_index": edge_index,
        "edge_feat": None,
        "node_feat": node_feat,
        "num_nodes": num_nodes,
    }
    dataset.label = label
    return dataset


def load_fixed_splits(dataset, sub_dataset):
    """
    loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != "None":
        name += f"-{sub_dataset}"

    if not os.path.exists(f"./splits/{name}-splits.npy"):
        assert dataset in SPLITS_DRIVE_URL.keys()
        gdd.download_file_from_google_drive(
            file_id=SPLITS_DRIVE_URL[dataset],
            dest_path=f"./splits/{name}-splits.npy",
            showsize=True,
        )

    splits_lst = np.load(f"./splits/{name}-splits.npy", allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


def load_full_data(dataset_name):
    if dataset_name in {"cora", "citeseer", "pubmed"}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
    elif args.dataset == 'reddit':
        return load_reddit_data(dataset, normalization, cuda)
    elif dataset_name == "deezer-europe":
        dataset = load_deezer_dataset()
        dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])
        row, col = dataset.graph["edge_index"]
        N = dataset.graph["num_nodes"]
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph["node_feat"], dataset.label

    else:
        graph_adjacency_list_file_path = os.path.join(
            "../new_data", dataset_name, "out1_graph_edges.txt"
        )
        graph_node_features_and_labels_file_path = os.path.join(
            "../new_data", dataset_name, "out1_node_feature_label.txt"
        )

        G = nx.DiGraph().to_undirected()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == "film":
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(","), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(","), dtype=np.uint8
                    )
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in G:
                    G.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in G:
                    G.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))

        features = np.array(
            [
                features
                for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])
            ]
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])]
        )

    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # .to(device)
    return adj, features, labels


def normalize(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1 / eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    else:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def random_disassortative_splits(labels, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])
    # print(train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = (1.0 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, ignore_negative=True):
    """
    randomly splits label into train/valid/test splits
    """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm="l1", axis=1)
    return sp.coo_matrix(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape,
        dtype=torch.float32,  # 明确指定数据类型
        device=values.device  # 保持与原数据相同的设备
    )


def train_model(
    model,
    optimizer,
    adj_low,
    adj_high,
    adj_low_unnormalized,
    features,
    labels,
    idx_train,
    criterion,
    dataset_name,
):
    model.train()
    optimizer.zero_grad()
    output, emb = model(features, adj_low, adj_high, adj_low_unnormalized)
    if dataset_name == "deezer-europe":
        output = F.log_softmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels.squeeze(1)[idx_train])
        acc_train = eval_acc(labels[idx_train], output[idx_train])
    else:
        output = F.log_softmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(labels[idx_train], output[idx_train])

    loss_train.backward()
    optimizer.step()

    return 100 * acc_train.item(), loss_train.item()


def train_prep(logger, args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    split_idx_lst = None

    # Training settings
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # # torch.backends.cudnn.deterministic = True
    # # torch.backends.cudnn.benchmark = False
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # # torch.backends.cudnn.benchmark = False
    # # torch.backends.cudnn.enabled = False
    # # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    # # os.environ['PYTHONHASHSEED'] = str(args.seed) 
    # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)   
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run info
    # model_info = {
    #     "model": args.model,
    #     "structure_info": args.structure_info,
    #     "dataset_name": args.dataset_name,
    #     "hidden": args.hidden,
    #     "init_layers_X": args.link_init_layers_X,
    #     "variant": args.variant,
    #     "layers": args.layers,
    #     "hop": args.hops,
    # }

    model_info = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "structure_info": args.structure_info,
        "init_layers_X": args.link_init_layers_X,
        "hidden": args.hidden,
        "layers": args.layers,
        "hops": args.hops,
        "fuse_hop": args.fuse_hop,
        "resnet": args.resnet,
        "layer_norm": args.layer_norm,
        "att_hopwise_distinct": args.att_hopwise_distinct,
        "optimizer": args.optimizer,
        "early_stopping": args.early_stopping,
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "dropout": args.dropout,
        "param_tunning": args.param_tunning,
        "no_cuda": args.no_cuda,
        "seed": args.seed,
        "num_splits": args.num_splits,
        "fixed_splits": args.fixed_splits,
        "normalization": args.normalization,
        "dataset_name": args.dataset_name,
        "online_cs": args.online_cs,
        "lambda_pen": args.lambda_pen,
        "lambda_2hop": args.lambda_2hop,
        "threshold": args.threshold,
        "comm_size": args.comm_size,
        # etc. You can store everything or just the subset you need.
    }

    run_info = {
        "result": 0,
        "std": 0,
        "lr": None,
        "weight_decay": None,
        "dropout": None,
        "runtime_average": None,
        "epoch_average": None,
        "split": None,
    }


    logger.log_init("Done Proccessing...")
    adj_low_unnormalized, features, labels = load_full_data(args.dataset_name)
    nnodes = labels.shape[0]
    I = torch.eye(nnodes).to(device)
    if (args.model == "acmgcnp" or args.model == "acmgcnpp" or args.model == "acmsmn") and (
        args.structure_info == 1
    ):
        pass
    else:
        features = normalize_tensor(features)

    if args.structure_info:
        adj_low = normalize_tensor(I + adj_low_unnormalized.to_dense().to(device))
        # adj_low = normalize_tensor(adj_low_unnormalized.to_dense())
        adj_high = (I - adj_low).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = adj_low_unnormalized.to(device)
    else:
        adj_low = normalize_tensor(I + adj_low_unnormalized.to_dense().to(device))
        # adj_low = normalize_tensor(adj_low_unnormalized.to_dense())
        adj_high = (I - adj_low).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = None

    if (args.model == "acmsgc") and (args.hops > 1):
        A_EXP = adj_low.to_dense()
        for _ in range(args.hops - 1):
            A_EXP = torch.mm(A_EXP, adj_low.to_dense())
        adj_low = A_EXP.to_sparse()
        del A_EXP
        adj_low = adj_low.to(device).to_sparse()

        # # high_freq
        # A_EXP = adj_high.to_dense()
        # for _ in range(args.hops - 1):
        #     A_EXP = torch.mm(A_EXP, adj_high.to_dense())
        # adj_high = A_EXP.to_sparse()
        # del A_EXP
        # adj_high = adj_high.to(device).to_sparse()


    if (args.model == "acmsmn") and (args.hops > 1):           
        # adj_low = adj_low.to_sparse()
        # adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).float()
        current_A_EXP_low = adj_low
        # A_EXP_high = adj_high.to_dense()
        # I = I.to_sparse()
        low_channels = [I, (adj_low - I)]
        high_channels = [I, (- I + adj_high)]
        unnormalized_channels = [I, adj_low]
        prev_A_EXP_low = adj_low
        for i in range(1, args.hops-1):
            current_A_EXP_low = torch.spmm(current_A_EXP_low, adj_low)
            distinct_A_EXP_low = torch.nn.functional.relu(current_A_EXP_low - prev_A_EXP_low - I)
            # distinct_A_EXP_low = (prev_A_EXP_low == 0).float() * current_A_EXP_low


            # degree = torch.diag(distinct_A_EXP_low.sum(dim=1))  # Compute degree matrix
            # D_inv = torch.linalg.inv(degree + 1e-9 * torch.eye(degree.size(0)).to(device))  # Add epsilon to avoid singularity
            # distinct_A_EXP_low = torch.mm(D_inv, distinct_A_EXP_low)

            # distinct_A_EXP_low = torch.nn.functional.normalize(distinct_A_EXP_low, p='fro', dim=1)

            # distinct_A_EXP_low = torch.nn.functional.softmax(distinct_A_EXP_low, dim=1)
            # distinct_A_EXP_high = I - distinct_A_EXP_low

            if args.normalization == 'softmax':
                distinct_A_EXP_low = torch.nn.functional.softmax(distinct_A_EXP_low, dim=1)
                distinct_A_EXP_high = I - distinct_A_EXP_low
            elif args.normalization == 'row_sum':
                row_sums = distinct_A_EXP_low.sum(dim=1, keepdim=True)
                distinct_A_EXP_low = distinct_A_EXP_low / (row_sums + 1e-9)  # Add epsilon to avoid division by zero
                distinct_A_EXP_high = I - distinct_A_EXP_low
            else:
                distinct_A_EXP_high = I - distinct_A_EXP_low
                
            low_channels.append(distinct_A_EXP_low)
            high_channels.append(distinct_A_EXP_high)
            unnormalized_channels.append(current_A_EXP_low)

            prev_A_EXP_low = current_A_EXP_low # distinct_A_EXP_high #   #######!!!!!!!!!!!!! distinct_A_EXP_low

        del current_A_EXP_low
        del distinct_A_EXP_low
        del distinct_A_EXP_high
        low_channels = torch.stack(low_channels, dim=0)
        adj_low = low_channels.to(device)
        high_channels = torch.stack(high_channels, dim=0)
        adj_high = high_channels.to(device)
        unnormalized_channels = torch.stack(unnormalized_channels, dim=0)
        adj_low_unnormalized = unnormalized_channels.to(device)


    # if (args.model == "acmsmn") and (args.hops > 1):        
    #     A_EXP_low = adj_low.to_dense()
    #     A_EXP_high = adj_high.to_dense()
    #     low_channels = [I]
    #     high_channels = [I]
    #     for i in range(args.hops-1):
    #         A_EXP_low = torch.mm(A_EXP_low, adj_low.to_dense())
    #         A_EXP_high = I - A_EXP_low
    #         low_channels.append(A_EXP_low)
    #         high_channels.append(A_EXP_high)
    #     del A_EXP_low
    #     del A_EXP_high
    #     low_channels = torch.stack(low_channels, dim=0).to_sparse()
    #     adj_low = low_channels.to(device)
    #     high_channels = torch.stack(high_channels, dim=0).to_sparse()
    #     adj_high = high_channels.to(device)


    if args.dataset_name == "deezer-europe":
        args.num_splits = 5
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(device)
        split_idx_lst = load_fixed_splits(args.dataset_name, "")

    return (
        device,
        model_info,
        run_info,
        adj_high,
        adj_low,
        adj_low_unnormalized,
        features,
        labels,
        split_idx_lst,
    )



# ------------------------------------------------------#


def cs_eval_metrics(communities, query_nodes, labels):
    true = [1] * (len([element for sublist in communities for element in sublist]) - len(query_nodes))
    pred = []
    for community , query in zip(communities, query_nodes):
        pred.append(np.equal(labels[torch.LongTensor(community[1:])].tolist(), int(labels[query])) * 1)

    pred = [element for sublist in pred for element in sublist]
    # acc = accuracy_score(true, pred).round(4)
    jaccard = round(jaccard_score(true, pred, average='binary'), 4)
    nmi = round(normalized_mutual_info_score(true, pred, average_method='arithmetic'), 4)
    f1 = f1_score(true, pred, average='binary').round(4)
    return f1, jaccard, nmi


def centroid_distance(community, features, prev_centroid=None, prev_node=None, new_node=None, community_size = 20):
    features_c = features[torch.LongTensor(community)]
    
    if prev_centroid is not None and prev_node is not None and new_node is not None:
        # If prev_centroid is 2D, squeeze it for arithmetic.
        prev_centroid_1d = prev_centroid.squeeze(0) if prev_centroid.dim() == 2 else prev_centroid
        updated_centroid = prev_centroid_1d - (1 / community_size) * (features[prev_node] - features[new_node])
    else:
        updated_centroid = torch.sum(features_c, dim=0) / community_size

    # Unsqueeze the centroid so that it has shape (1, feature_dim) for cosine similarity.
    updated_centroid = updated_centroid.unsqueeze(0)
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(features_c, updated_centroid.expand_as(features_c))
    return similarity, updated_centroid  # You can choose to squeeze updated_centroid if needed


def sub_cs(features, query_nodes, community_size, early_stop):
    communities = []
    cs_start = perf_counter()

    for query in query_nodes:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = cos(features, features[query].reshape(1, -1)).squeeze()
        topk_prob, topk_idx = torch.topk(cos_simi, community_size * 2)
        
        community = topk_idx[:community_size].tolist()
        # Pass community_size in the initial call
        similarity, centroid = centroid_distance(community, features, community_size=community_size)
        min_simi, min_idx = torch.min(similarity, dim=0)
        least_similar_node = community[min_idx]

        for idx in topk_idx[community_size:]:
            query_simi = cos(features[idx].reshape(1, -1), centroid)
            if query_simi > min_simi:
                community[min_idx] = idx.item()
                similarity, centroid = centroid_distance(community, features, centroid, least_similar_node, idx, community_size)
                min_simi, min_idx = torch.min(similarity, dim=0)
                least_similar_node = community[min_idx]
                if least_similar_node == query:
                    break
        
        communities.append(community)
    
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    return communities, cs_time

# def objective_function_2hop(community, features, query_node, adj, adj_2, 
#                               threshold=0.2, lambda_pen=1.0, lambda_2hop=0.5):
#     """
#     Compute the multi-objective value for a candidate community.

#     The objective is defined as:
#       Objective = AvgSim(q, C) + λ₂hop * AvgBonus(C) - λ_pen * AvgPenalty(C)
      
#     where:
#       - AvgSim(q, C) is the average cosine similarity between the query node and each node in C.
#       - AvgPenalty(C) is the average shortfall for each directly connected pair in C whose cosine similarity
#         is below the threshold.
#       - AvgBonus(C) is the average bonus over node pairs that are not directly connected but whose cosine similarity
#         is above the threshold (reflecting informative 2-hop connectivity).

#     Args:
#         community (list): List of node indices forming the candidate community.
#         features (torch.Tensor): Node embeddings, shape [num_nodes, feature_dim].
#         query_node (int): Index of the query node.
#         adj (torch.Tensor or np.ndarray): Unsigned (binary) adjacency matrix. Nonzero indicates a direct edge.
#         adj_2 (torch.Tensor or np.ndarray): Precomputed 2-hop connectivity matrix.
#         threshold (float): Cosine similarity threshold.
#         lambda_pen (float): Weight for the penalty term.
#         lambda_2hop (float): Weight for the 2-hop bonus term.
        
#     Returns:
#         float: The computed objective value.
#     """
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
#     # 1. Cohesiveness: average cosine similarity between the query node and community nodes.
#     sims = []
#     for node in community:
#         sim = F.cosine_similarity(features[query_node].unsqueeze(0), features[node].unsqueeze(0), dim=1, eps=1e-6).item()
#         sims.append(sim)
#     avg_sim = np.mean(sims) if sims else 0.0

#     # 2. Penalty: for directly connected pairs whose cosine similarity is below threshold.
#     penalty = 0.0
#     pen_count = 0
#     for i in range(len(community)):
#         for j in range(i+1, len(community)):
#             u = community[i]
#             v = community[j]
#             # Only consider pairs that are directly connected.
#             if adj[u, v] != 0 or adj[v, u] != 0:
#                 sim = F.cosine_similarity(features[u].unsqueeze(0), features[v].unsqueeze(0), dim=1, eps=1e-6).item()
#                 if sim < threshold:
#                     penalty += (threshold - sim)
#                     pen_count += 1
#     avg_penalty = penalty / pen_count if pen_count > 0 else 0.0

#     # 3. 2-hop bonus: for node pairs that are NOT directly connected, add bonus if their cosine similarity > threshold.
#     bonus = 0.0
#     bonus_count = 0
#     for i in range(len(community)):
#         for j in range(i+1, len(community)):
#             u = community[i]
#             v = community[j]
#             # Only consider pairs that are not directly connected.
#             if adj[u, v] == 0 and adj[v, u] == 0:
#                 sim = F.cosine_similarity(features[u].unsqueeze(0), features[v].unsqueeze(0), dim=1, eps=1e-6).item()
#                 if sim > threshold:
#                     bonus += adj_2[u, v]  # Use the 2-hop connectivity value as bonus.
#                     bonus_count += 1
#     avg_bonus = bonus / bonus_count if bonus_count > 0 else 0.0

#     # Overall objective.
#     objective = avg_sim + lambda_2hop * avg_bonus - lambda_pen * avg_penalty
#     return objective

# def signed_cs(adj, adj_2, features, query_nodes, community_size, early_stop,
#                                 threshold=0.2, lambda_pen=1.0, lambda_2hop=0.5):
#     """
#     Perform community search for each query node as follows:
#       1. Compute the top (community_size * 2) most similar nodes to the query.
#       2. Use these as an initial candidate community.
#       3. Iteratively remove one node at a time (choosing the removal that maximizes
#          the objective function) until the community size is reduced to 'community_size'.
    
#     Args:
#         adj (torch.Tensor or np.ndarray): Unsigned (binary) adjacency matrix.
#         adj_2 (torch.Tensor or np.ndarray): Precomputed 2-hop connectivity matrix.
#         features (torch.Tensor): Node embeddings.
#         query_nodes (iterable): Iterable of query node indices.
#         community_size (int): Desired final community size (k).
#         early_stop (int): Early stopping parameter (not used in this example).
#         threshold (float): Cosine similarity threshold.
#         lambda_pen (float): Weight for the penalty term.
#         lambda_2hop (float): Weight for the 2-hop bonus term.
    
#     Returns:
#         tuple: (communities, cs_time) where communities is a list of communities (each a list of node indices)
#                and cs_time is the average time per query.
#     """
#     communities = []
#     cs_start = perf_counter()
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
#     for query in query_nodes:
#         # Compute cosine similarities from query to all nodes.
#         cos_simi = cos(features, features[query].reshape(1, -1)).squeeze()
#         # Get the top (community_size * 2) candidates.
#         _, top2k_idx = torch.topk(cos_simi, community_size * 2)
#         candidate_community = top2k_idx.tolist()  # initial candidate set of size 2k
        
#         # Compute current objective value.
#         current_obj = objective_function_2hop(candidate_community, features, query, adj, adj_2, threshold, lambda_pen, lambda_2hop)
        
#         # Iteratively remove nodes until the community has size 'community_size'.
#         while len(candidate_community) > community_size:
#             best_obj = -float('inf')
#             best_removal = None
#             # Try removing each node in the candidate set.
#             for node in candidate_community:
#                 new_community = candidate_community.copy()
#                 new_community.remove(node)
#                 obj = objective_function_2hop(new_community, features, query, adj, adj_2, threshold, lambda_pen, lambda_2hop)
#                 if obj > best_obj:
#                     best_obj = obj
#                     best_removal = node
#             # If removal improves the objective, update the candidate set.
#             if best_obj > current_obj:
#                 candidate_community.remove(best_removal)
#                 current_obj = best_obj
#             else:
#                 # If no removal improves the objective, break out of the loop.
#                 break
#         # If the candidate community is still larger than desired, trim arbitrarily.
#         candidate_community = candidate_community[:community_size]
#         communities.append(candidate_community)
    
#     cs_time = (perf_counter() - cs_start) / len(query_nodes)
#     return communities, cs_time


def community_objective(candidate, S):
    """
    Sum of S[u,v] over all pairs (u,v) in 'candidate' (u < v).
    """
    score = 0.0
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            u = candidate[i]
            v = candidate[j]
            score += S[u, v]
    return score

import torch

def community_objective_extend(candidate, S, adj, adj_2, 
                        threshold=0.2, 
                        lambda_pen=1.0, 
                        lambda_2hop=0.5):
    """
    Compute the objective for a candidate community:
    
      BaseScore = sum of S[u,v] over all (u,v) in the community
      Penalty   = sum of (threshold - S[u,v]) for direct edges with S[u,v] < threshold
      Bonus     = sum of adj_2[u,v] for pairs that are:
                   - not directly connected in 'adj'
                   - S[u,v] > threshold
                   - adj_2[u,v] > 0
      Overall   = BaseScore + lambda_2hop * Bonus - lambda_pen * Penalty
    
    Args:
      candidate (list): nodes in the candidate community
      S (Tensor or array): signed adjacency (NxN)
      adj (Tensor or array): binary adjacency (NxN)
      adj_2 (Tensor or array): 2-hop adjacency (NxN)
      threshold (float): cutoff for penalizing edges or rewarding 2-hop pairs
      lambda_pen (float): penalty weight
      lambda_2hop (float): bonus weight
    
    Returns:
      float: the final objective value
    """
    base_score = 0.0
    penalty_sum = 0.0
    bonus_sum = 0.0
    
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            u = candidate[i]
            v = candidate[j]
            
            # Retrieve S[u,v]
            s_val = S[u, v]
            # If S is a torch Tensor, convert to a Python float
            if isinstance(s_val, torch.Tensor):
                s_val = s_val.item()
            
            base_score += s_val
            
            # Check if there's a direct edge => potential penalty
            # We assume 'adj[u,v] == 1' means direct neighbor
            if adj[u, v] != 0:
                # If S[u,v] is below threshold, impose a penalty
                if s_val < threshold:
                    penalty_sum += (threshold - s_val)
            else:
                # If not directly connected => potential 2-hop bonus
                # Only add bonus if adj_2[u,v] > 0 and s_val > threshold
                if adj_2[u, v] > 0 and s_val > threshold:
                    bonus_sum += adj_2[u, v]
    
    # Combine base, bonus, penalty
    return base_score + lambda_2hop * bonus_sum - lambda_pen * penalty_sum


def signed_cs(
    S, 
    query_nodes, 
    emb=None,
    community_size=10, 
    top_factor=2,
    threshold=0.2,
    lambda_pen=1.0,
    lambda_2hop=0.5,
    adj=None,
    adj_2=None
):
    """
    Signed-graph-based community search:
      - S: signed adjacency (NxN).
      - query_nodes: list or tensor of node indices to query.
      - emb: node embeddings (not strictly necessary here, but might be used).
      - community_size: desired final k
      - top_factor: we pick top_factor*k initial candidates by sim to query.

    Returns a list of communities (each is a list of node IDs).
    """
    cs_start = perf_counter()
    communities = []
    N = S.shape[0]
    
    # Precompute node similarities for each query => pick top 2k
    # (assuming we already have sim_matrix or we can reuse S + threshold)
    # But let's do a direct approach: we can derive sim from S + threshold if needed
    # sim[u,v] = S[u,v] + threshold
    # so for each query, sim[q,u] = S[q,u] + threshold.
    
    for q in query_nodes:
        # (A) Build initial candidate => top 2k by sim to q
        # sim[q,u] = S[q,u] + threshold
        # let's reconstruct "sim" on the fly
        sim_q = S[q,:]  # shape [N], but we need to add threshold if we want actual similarity
        # just do: actual_sim_q[u] = S[q,u] + TH
        # we'll pick top2k based on that.
        # But we need the threshold used in building S. Let's guess we stored it or pass it in.
        # For simplicity, let's just pick top2k by S[q,u], i.e. the highest S-values.
        # That means we pick the nodes that produce the largest "margin above threshold."
        top_vals, top_indices = torch.topk(sim_q, community_size * top_factor)
        candidate = top_indices.tolist()
        
        # remove 'q' if you want to keep the query node in or out. 
        # Typically we keep query in the set:
        if q not in candidate:
            candidate[-1] = q  # ensure query is in the set
        # current_obj = community_objective(candidate, S)
        current_obj = community_objective_extend(candidate, S, adj, adj_2,
                                                threshold, lambda_pen, lambda_2hop)
        
        # (B) Iteratively remove nodes until size == community_size
        while len(candidate) > community_size:
            best_obj = -1e15
            best_node = None
            for node in candidate:
                new_cand = [x for x in candidate if x != node]
                # test_obj = community_objective(new_cand, S)
                test_obj = community_objective_extend(new_cand, S, adj, adj_2,
                                                      threshold, lambda_pen, lambda_2hop)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_node = node
            # If best removal improves objective, accept it; otherwise stop
            if best_obj > current_obj:
                candidate.remove(best_node)
                current_obj = best_obj
            else:
                break
        
        # (C) Final candidate
        if len(candidate) > community_size:
            candidate = candidate[:community_size]
        communities.append(candidate)
    cs_time = (perf_counter() - cs_start) / len(query_nodes)

    return communities, cs_time

def community_objective_from_sums(base_sum, penalty_sum, bonus_sum, 
                                  lambda_pen, lambda_2hop):
    """
    Objective = base_sum + lambda_2hop * bonus_sum - lambda_pen * penalty_sum
    """
    return base_sum + lambda_2hop * bonus_sum - lambda_pen * penalty_sum

def signed_cs_fast(
    S,
    query_nodes,
    adj,
    adj_2,
    threshold=0.2,
    lambda_pen=1.0,
    lambda_2hop=0.5,
    community_size=10,
    top_factor=2
):
    """
    A faster (partial-sum) version of the signed CS:
      - 'S': signed adjacency, shape [N, N], S[u,v] = sim(u,v) - threshold0 (some reference).
      - 'adj': binary adjacency (N, N)
      - 'adj_2': 2-hop adjacency (N, N)
      - threshold: used for penalty/bonus checks
      - lambda_pen, lambda_2hop: weights for penalty/bonus
      - community_size: desired final size
      - top_factor: build initial candidate of size 'top_factor*k'
      
    We'll do an iterative "remove one node" local search, but 
    with partial sums for base-score, penalty, and bonus.
    """
    t0 = perf_counter()
    communities = []
    
    # We'll check if S, adj, adj_2 are torch Tensors or numpy arrays
    use_torch = isinstance(S, torch.Tensor)
    
    N = S.shape[0]
    
    # A small helper to fetch S[u,v], adj[u,v], adj_2[u,v] as float
    # (handles torch or numpy indexing)
    def get_val(matrix, u, v):
        val = matrix[u, v]
        if isinstance(val, torch.Tensor):
            return val.item()
        return val
    
    # =============== Main Loop over queries =================
    for q in query_nodes:
        # --- (A) Build initial candidate using top-k from S[q]
        if use_torch:
            sim_q = S[q]  # shape [N]
            top_vals, top_idx = torch.topk(sim_q, community_size * top_factor)
            candidate = top_idx.tolist()
        else:
            sim_q = S[q]
            candidate = np.argsort(-sim_q)[:community_size * top_factor].tolist()
        
        # Ensure query is in the community
        if q not in candidate:
            candidate[-1] = q
        
        # Make a set for quick membership checks
        cand_set = set(candidate)
        
        # --- (B) Compute partial sums once for the entire candidate
        base_sum = 0.0
        penalty_sum = 0.0
        bonus_sum = 0.0
        
        # We'll store for each node: base_contrib[node], penalty_contrib[node], bonus_contrib[node]
        base_contrib = {}
        penalty_contrib = {}
        bonus_contrib = {}
        
        # fill them in
        for u in candidate:
            bc = 0.0
            pc = 0.0
            bnc = 0.0
            for v in candidate:
                if v == u:
                    continue
                s_uv = get_val(S, u, v)
                
                bc += s_uv
                
                # Check adjacency
                if get_val(adj, u, v) != 0:
                    # direct edge => potential penalty if s_uv < threshold
                    if s_uv < threshold:
                        pc += (threshold - s_uv)
                else:
                    # not directly connected => potential 2-hop bonus
                    if get_val(adj_2, u, v) > 0 and s_uv > threshold:
                        bnc += get_val(adj_2, u, v)
            
            base_contrib[u] = bc
            penalty_contrib[u] = pc
            bonus_contrib[u] = bnc
            
            # We'll add them to global sums,
            # but remember each pair is counted from both sides => divide by 2 after
            base_sum += bc
            penalty_sum += pc
            bonus_sum += bnc
        
        # Now each pair is counted from both ends
        base_sum    *= 0.5
        penalty_sum *= 0.5
        bonus_sum   *= 0.5
        
        # current objective
        current_obj = community_objective_from_sums(base_sum, penalty_sum, bonus_sum,
                                                    lambda_pen, lambda_2hop)
        
        # --- (C) Iteratively remove nodes until community_size
        while len(candidate) > community_size:
            best_obj = -1e15
            best_node = None
            
            # Try removing each node
            for node in candidate:
                # removing 'node' => new sums:
                # Subtract that node's half from global
                # But we have to do: new_base_sum = base_sum - base_contrib[node]
                # Then also update because we are double-counting pairs. 
                new_base = base_sum - (base_contrib[node] * 0.5)
                new_pen = penalty_sum - (penalty_contrib[node] * 0.5)
                new_bon = bonus_sum - (bonus_contrib[node] * 0.5)
                
                # Why 0.5? Because each contribution was counted from both endpoints.
                # Actually, we've already halved the global sums. 
                # 'base_contrib[node]' was the sum from node's perspective, 
                # so removing it from the global sum that was halved => multiply by 0.5.
                
                # test objective
                test_obj = community_objective_from_sums(new_base, new_pen, new_bon,
                                                         lambda_pen, lambda_2hop)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_node = node
            
            if best_obj > current_obj and best_node is not None:
                # => remove best_node for real
                r = best_node
                candidate.remove(r)
                cand_set.remove(r)
                
                # Now we update the global sums
                # Because we are removing r entirely
                base_sum  -= (base_contrib[r] * 0.5)
                penalty_sum -= (penalty_contrib[r] * 0.5)
                bonus_sum   -= (bonus_contrib[r] * 0.5)
                
                # We must also update the partial contributions of the rest of candidate
                # because they no longer pair with 'r'.
                for other in candidate:
                    s_ov = get_val(S, other, r)
                    
                    # remove s_ov from other.base_contrib
                    base_contrib[other] -= s_ov
                    
                    # check if it was a penalty or bonus
                    if get_val(adj, other, r) != 0:
                        # direct edge => if s_ov<threshold => remove that penalty
                        if s_ov < threshold:
                            pen_val = (threshold - s_ov)
                            penalty_contrib[other] -= pen_val
                    else:
                        # not direct => if s_ov>threshold and adj_2>0 => remove that bonus
                        if get_val(adj_2, other, r) > 0 and s_ov > threshold:
                            bon_val = get_val(adj_2, other, r)
                            bonus_contrib[other] -= bon_val
                
                # done: removing r also means we won't need those partials
                del base_contrib[r]
                del penalty_contrib[r]
                del bonus_contrib[r]
                
                # update current_obj
                current_obj = best_obj
            else:
                break  # no improvement => done
        
        # if we still have more than needed, trim
        if len(candidate) > community_size:
            candidate = candidate[:community_size]
        
        communities.append(candidate)
    
    total_time = perf_counter() - t0
    avg_time = total_time / len(query_nodes) if len(query_nodes) else 0.0
    return communities, avg_time



def precompute_neighbor_lists(adj, adj_2):
    """预构建邻接表和两跳邻居表（兼容PyTorch和NumPy）"""
    N = adj.shape[0]
    neighbors = {}
    two_hop_neighbors = {}

    for u in range(N):
        if isinstance(adj, torch.Tensor):
            neighbors_u = torch.nonzero(adj[u] > 0).flatten().tolist()
            two_hop_u = torch.nonzero(adj_2[u] > 0).flatten().tolist()
        else:
            neighbors_u = np.where(adj[u] > 0)[0].tolist()
            two_hop_u = np.where(adj_2[u] > 0)[0].tolist()

        neighbors[u] = set(neighbors_u)
        two_hop_neighbors[u] = set(two_hop_u)

    return neighbors, two_hop_neighbors


def signed_cs_optimized(
    S,
    query_nodes,
    adj,
    adj_2,
    threshold=0.2,
    lambda_pen=1.0,
    lambda_2hop=0.5,
    community_size=20,
    top_factor=20,
    use_heap=True,
    early_pruning=True,
    sample_size=100
):
    t0 = perf_counter()
    communities = []
    use_torch = isinstance(S, torch.Tensor)
    neighbors, two_hop_neighbors = precompute_neighbor_lists(adj, adj_2)

    def get_val(matrix, u, v):
        val = matrix[u, v]
        return val.item() if isinstance(val, torch.Tensor) else val

    for q in query_nodes:
        sim_q = S[q]
        candidate = (
            torch.topk(sim_q, community_size * top_factor).indices.tolist()
            if use_torch else np.argsort(-sim_q)[:community_size * top_factor].tolist()
        )

        if early_pruning:
            median_sim = torch.median(sim_q).item() if use_torch else np.median(sim_q)
            candidate = [u for u in candidate if get_val(S, q, u) >= median_sim]

        if q not in candidate:
            candidate.append(q)

        candidate = list(set(candidate))
        if not candidate:
            candidate = [q]
        
        base_contrib = {}
        penalty_contrib = {}
        bonus_contrib = {}

        base_sum = 0.0
        penalty_sum = 0.0
        bonus_sum = 0.0

        for u in candidate:
            base_contrib[u] = sum(get_val(S, u, v) for v in candidate if u != v)
            penalized = [v for v in neighbors[u] if v in candidate and get_val(S, u, v) < threshold]
            pen_val = sum(threshold - get_val(S, u, v) for v in penalized)
            penalty_contrib[u] = pen_val

            bonused = [v for v in two_hop_neighbors[u] if v in candidate and get_val(S, u, v) > threshold]
            bon_val = sum(get_val(adj_2, u, v) for v in bonused)
            bonus_contrib[u] = bon_val

            base_sum += base_contrib[u]
            penalty_sum += pen_val
            bonus_sum += bon_val

        base_sum /= 2
        penalty_sum /= 2
        bonus_sum /= 2

        def calculate_removal_gain(u):
            return base_contrib[u] / 2 + lambda_pen * penalty_contrib[u] / 2 - lambda_2hop * bonus_contrib[u] / 2

        if use_heap:
            heap = [(-calculate_removal_gain(u), u) for u in candidate]
            heapq.heapify(heap)

        current_obj = base_sum - lambda_pen * penalty_sum + lambda_2hop * bonus_sum

        while len(candidate) > community_size:
            if sample_size and len(candidate) > sample_size:
                candidates_to_try = np.random.choice(candidate, size=sample_size, replace=False)
                best_node = max(candidates_to_try, key=calculate_removal_gain)
            elif use_heap:
                if not heap:
                    break
                delta, best_node = heapq.heappop(heap)
                if -delta <= 0:
                    break
            else:
                best_node = max(candidate, key=calculate_removal_gain)

            candidate.remove(best_node)
            affected_nodes = (neighbors[best_node] | two_hop_neighbors[best_node]) & set(candidate)

            for u in affected_nodes:
                s_uv = get_val(S, u, best_node)
                base_sum -= s_uv / 2
                base_contrib[u] -= s_uv

                if u in neighbors[best_node] and s_uv < threshold:
                    penalty_sum -= (threshold - s_uv) / 2
                    penalty_contrib[u] -= threshold - s_uv

                if best_node in two_hop_neighbors[u] and s_uv > threshold:
                    adj2_val = get_val(adj_2, u, best_node)
                    bonus_sum -= adj2_val / 2
                    bonus_contrib[u] -= adj2_val

            current_obj = base_sum - lambda_pen * penalty_sum + lambda_2hop * bonus_sum

            if use_heap:
                heap = [(-calculate_removal_gain(u), u) for u in candidate]
                heapq.heapify(heap)

        communities.append(candidate[:community_size])

    avg_time = (perf_counter() - t0) / len(query_nodes) if len(query_nodes) > 0 else 0

    return communities, avg_time


def build_signed_adjacency(emb, threshold):
    """
    1) emb is shape [N, d].
    2) We compute S[u,v] = cos_sim(u,v) - threshold.
       where cos_sim(u,v) = (normed_emb[u] dot normed_emb[v]).
    3) Return S as a torch.Tensor (NxN) on the same device as emb.
    """
    device = emb.device
    N = emb.size(0)
    
    # Normalize => dot product = cosine similarity
    norms = emb.norm(dim=1, keepdim=True)
    normed_emb = emb / (norms + 1e-12)
    sim_matrix = normed_emb @ normed_emb.t()  # shape [N, N]
    
    S = sim_matrix - threshold
    return S

def build_positive_graph_torch(S):
    """
    Build adjacency lists for edges where S[u,v] >= 0, 
    returning a Python list of lists (or sets):
       pos_graph[u] = [v1, v2, ...] for all v with S[u,v]>=0, v != u
    
    S must be a 2D torch.Tensor on any device (CPU or GPU).
    We'll move its indices to CPU as we create adjacency in Python.
    """
    # 1) Find all (u,v) with S[u,v]>=0
    #    'torch.where' returns row_idx, col_idx
    row_idx, col_idx = torch.where(S >= 0)
    
    # 2) Convert to Python adjacency structure
    #    We'll store adjacency in a python list of lists, one for each row.
    N = S.size(0)
    pos_graph = [[] for _ in range(N)]
    
    # row_idx[i], col_idx[i] are matched pairs where S[u,v]>=0
    # We'll accumulate them into pos_graph[u].
    # Important: We do '.tolist()' or '.item()' to convert the scalar Tensors.
    # Because BFS in Python requires standard int indices.
    for i in range(len(row_idx)):
        u = row_idx[i].item()
        v = col_idx[i].item()
        if u != v:
            pos_graph[u].append(v)
    
    return pos_graph

def bfs_with_teleport(pos_graph, query, sim_q, k):
    """
    pos_graph[u]: list of neighbors v where S[u,v]>=0
    query (int): index of query node
    sim_q (torch.Tensor or array-like): similarity from query to all nodes
    k (int): target community size
    
    We'll do BFS from 'query'. If BFS ends early and we have < k nodes, 
    we teleport to the unvisited node with the highest sim_q, etc.
    """
    from collections import deque
    
    visited = set([query])
    queue = deque([query])
    N = len(pos_graph)
    
    while True:
        # normal BFS until queue is empty or we reach k
        while queue and len(visited) < k:
            u = queue.popleft()
            for v in pos_graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        if len(visited) >= k:
            break  # done

        # If we are stuck (queue empty) but haven't reached k => teleport
        if not queue and len(visited) < k:
            # find best unvisited node by sim_q
            candidates = []
            for node_idx in range(N):
                if node_idx not in visited:
                    # sim_q[node_idx] could be a scalar tensor => to float
                    sim_val = float(sim_q[node_idx])
                    candidates.append((sim_val, node_idx))
            if not candidates:
                break  # no more nodes left
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_node = candidates[0][1]
            visited.add(best_node)
            queue.append(best_node)

        # If queue is empty, we may teleport again in the next loop iteration
        if not queue:
            pass
    
    return list(visited)


def bfs_teleport(emb, query_nodes, k, threshold=0.2):
    """
    1) Build signed adjacency from embeddings.
    2) Build positive graph pos_graph (S[u,v]>=0).
    3) For each query:
        - BFS from q
        - if BFS gets stuck < k, teleport to next best unvisited node
        - continue until k reached or no more nodes
    Returns a list of communities (each up to size k).
    """
    cs_start = perf_counter()

    # 1) Signed adjacency
    S = build_signed_adjacency(emb, threshold=threshold)
    
    # 2) Build adjacency list
    pos_graph = build_positive_graph_torch(S)
    
    # 3) For each query, run BFS+teleport
    communities = []
    
    # Also build sim_matrix if we need sim_q for each query:
    # Because BFS teleport picks "most similar to query"
    # we can do sim_q = sim_matrix[q], etc.
    with torch.no_grad():
        norms = emb.norm(dim=1, keepdim=True)
        normed_emb = emb / (norms + 1e-12)
        sim_matrix = normed_emb @ normed_emb.t()  # shape NxN

    for q in query_nodes:
        # sim_q => 1D vector of similarity from q to all nodes
        sim_q = sim_matrix[q]  # still a GPU tensor or CPU tensor
        comm = bfs_with_teleport(pos_graph, q, sim_q, k)
        
        # optional: place q at index 0 in the list
        if q in comm:
            comm.remove(q)
            comm.insert(0, q)
        communities.append(comm)
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    
    return communities, cs_time

def sub_topk(features, query_nodes, community_size, early_stop):
    """
    For each query node, select the top-k most similar nodes based on cosine similarity.
    
    Args:
        features (torch.Tensor): Node feature matrix.
        query_nodes (torch.Tensor): Tensor of query node indices.
        community_size (int): Number of nodes to select for each community.
        early_stop (int): Early stopping parameter (not used in this implementation).
    
    Returns:
        tuple: (communities, cs_time)
            communities: A list where each element is a list of node indices representing a community.
            cs_time: Average time taken per query.
    """
    communities = []
    cs_start = perf_counter()
    
    for query in query_nodes:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # Compute cosine similarity between the query node and all nodes.
        cos_simi = cos(features, features[query].reshape(1, -1)).squeeze()
        # Select the top-k similar nodes (community_size many).
        _, topk_idx = torch.topk(cos_simi, community_size)
        community = topk_idx.tolist()
        communities.append(community)
    
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    return communities, cs_time


def community_search(adj, adj_2, query_nodes, emb, community_size, labels, method='sub_cs'):
    """
    Perform community search given query nodes, embeddings, and adjacency matrix.
    
    Args:
        query_nodes (torch.LongTensor): Nodes to query.
        emb (torch.Tensor): Node embeddings.
        adj (torch.sparse.Tensor): Adjacency matrix.
        community_size (int): Size of the community to find.
        method (str): Community search method ('sub_cs', 'sub_topk', 'BFS').
    
    Returns:
        tuple: cs_f1, cs_jaccard, cs_nmi, cs_time
    """
    if method == 'sub_cs':
        communities, cs_time = sub_cs(emb, query_nodes, 
                                     community_size=community_size, early_stop=8)
    elif method == 'sub_topk':
        communities, cs_time = sub_topk(emb, query_nodes, 
                                       community_size=community_size, early_stop=8)
    elif method == 'signed_cs':
        norms = emb.norm(dim=1, keepdim=True)
        normed_feats = emb / (norms + 1e-12)
        sim_matrix = normed_feats @ normed_feats.T  # NxN
        S = sim_matrix - 0.2 # threshold
        communities, cs_time = signed_cs_fast(S,query_nodes,
                                                adj,
                                                adj_2,
                                                threshold=args.threshold,
                                                lambda_pen=args.lambda_pen,
                                                lambda_2hop=args.lambda_2hop,
                                                community_size=community_size,
                                                top_factor=2)  # how big is initial candidate set (2k by default))
    elif method == 'bfs_teleport':
        communities, cs_time = bfs_teleport(emb, query_nodes, community_size, args.threshold)
    else:
        raise ValueError("Invalid community search method")
    
    cs_f1, cs_jaccard, cs_nmi = cs_eval_metrics(communities, query_nodes, labels)
    
    return cs_f1, cs_jaccard, cs_nmi, cs_time