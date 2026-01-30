import os
from os import path
import gc
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
from scipy.fftpack import dct
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize as sk_normalize
from time import perf_counter
import torch
import torch.nn.functional as F
import torch_sparse
from arg_parser import arg_parser
from torch_geometric.utils import add_remaining_self_loops
from collections import deque
from typing import List, Sequence
import random

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

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="../data/"):
    t = perf_counter()
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("../data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test

    adj = adj + adj.T
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
   
    print(f"check point: reddit load in {perf_counter() - t:.3f} seconds")
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
    elif dataset_name == 'reddit':
        adj, features, labels = load_reddit_data(dataset_name)
        features = features.to_dense()
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
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_rw(adj: sp.spmatrix):
    """
    Random-Walk (row) normalization of an adjacency matrix.

    \tilde A = D^{-1} (A + I)
    where D_{ii} = sum_j (A_{ij} + I_{ij})
    """
    adj_self = adj + sp.eye(adj.shape[0], dtype=adj.dtype)
    adj_self = sp.coo_matrix(adj_self)
    rowsum = np.array(adj_self.sum(1)).flatten()          
    d_inv = np.reciprocal(rowsum, where=rowsum != 0)
    d_inv[rowsum == 0] = 0.0                              
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj_self).tocoo()


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
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
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
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_csr_tensor(sparse_mx):
    """
    Convert a scipy.sparse CSR matrix to a torch.sparse_csr_tensor.
    """
    if not sp.isspmatrix_csr(sparse_mx):
        sparse_mx = sparse_mx.tocsr()

    sparse_mx.sort_indices()  # Ensure indices are sorted (required by PyTorch)

    crow_indices = torch.from_numpy(sparse_mx.indptr.astype(np.int64))
    col_indices = torch.from_numpy(sparse_mx.indices.astype(np.int64))
    values = torch.from_numpy(sparse_mx.data.astype(np.float32))

    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=shape)


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
    output, emb, att = model(features, adj_low, adj_high, adj_low_unnormalized)
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

def dct_reduce(x: torch.Tensor, dim_half: int):
    x_np = x.detach().cpu().numpy()
    x_dct = dct(x_np, type=2, norm='ortho', axis=1) 
    x_dct = x_dct[:, :dim_half]  
    return torch.from_numpy(x_dct).to(x.device).to(x.dtype)

def cross_channel_skip_connect(args, features, adj_low, adj_high, I, device):

    t = perf_counter()
    if (args.model == "adaptcs") and (args.hops > 1):

        features = features.to(device)

        feat_low  = features  # [N, d]
        feat_high = features  # [N, d]

        low_channels_list  = [feat_low]
        high_channels_list = [feat_high]

        for hop_i in range(1, args.hops):
            
            D = feat_low.shape[1]
            half = D // 2
            feat_low_half = feat_low[:, :half]
            feat_high_half = feat_high[:, :D - half]
            in_combined = torch.cat([feat_low_half, feat_high_half], dim=1)


            print(f"check point: reddit precompute before spmm")
            next_low  = torch.spmm(adj_low,  in_combined)   # [N, 2d]
            next_high = torch.spmm(adj_high, in_combined)   # [N, 2d]

            low_channels_list.append(next_low)
            high_channels_list.append(next_high)

            feat_low  = next_low
            feat_high = next_high
            print(f"check point: reddit precompute hop {hop_i:.1f} done in {perf_counter() - t:.3f} seconds")

        print(f"check point: reddit all hop precompute done in {perf_counter() - t:.3f} seconds")

        gc.collect()
        torch.cuda.empty_cache()

        precompute_time = perf_counter() - t
        print(f"Precompute time: {precompute_time:.4f} seconds")

        return low_channels_list, high_channels_list

    else:
        return features, None


def torch_sparse_to_scipy(tensor_sp):

    tensor_sp = tensor_sp.coalesce().cpu()
    indices = tensor_sp.indices().numpy()  # shape [2, nnz]
    values  = tensor_sp.values().numpy()   # shape [nnz,]
    shape   = tensor_sp.size()             # (N, N)
    mat_scipy = sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    return mat_scipy


def distinct_hop_precompute_svds_low(args, adj_low, adj_high, I, features, device):

    t0 = perf_counter()

    A_sp = torch_sparse_to_scipy(adj_low)
    N = A_sp.shape[0]
    r = min(args.svd_rank, N)

    print(f"[SVDS] shape=({N},{N}), rank={r}")
    u_np, s_np, vt_np = svds(A_sp, k=r, which='LM')

    U  = torch.from_numpy(u_np.copy()).float().to(device)  
    S  = torch.from_numpy(s_np.copy()).float().to(device)   
    Vt = torch.from_numpy(vt_np.copy()).float().to(device)  

    VtX   = Vt @ features                            
    onesN = torch.ones(N, 1, device=device)          
    Vt1   = Vt @ onesN                                #

    low_features  = [features]        
    high_features = [features]
    adj_low_unnormalized = [adj_low]            

    for k in range(1, args.hops):
        S_k    = S ** k
        S_km1  = S ** (k - 1)
        S_diff = S_k - S_km1 - S                      

        Delta_diag   = torch.diag(S_diff)             
        distinct_low = U @ (Delta_diag @ VtX)       

        if getattr(args, "normalization", "").lower() == "global_w":
            row_sum = U @ (Delta_diag @ Vt1)         
            w       = torch.sigmoid(row_sum)          
            distinct_low_feat = w * distinct_low           

            distinct_high_feat = (1.0 - w) * distinct_low - features
        else:
            distinct_low_feat = distinct_low
            distinct_high_feat = features - distinct_low_feat

        low_features.append(distinct_low_feat)
        high_features.append(distinct_high_feat)

    # ------------------ cleanup --------------------------------------------
    del A_sp, U, S, Vt, u_np, s_np, vt_np, VtX, Vt1, onesN
    gc.collect(); torch.cuda.empty_cache()

    print(f"[distinct_hop_precompute_svds] time: {perf_counter()-t0:.4f}s")
    return low_features, high_features, adj_low_unnormalized


def distinct_hop_precompute_randomized_svds(args, adj_low, adj_high, I, features, device):
    """
    Compute distinct-hop features using randomized SVD on GPU.
    
    Args:
        args: argument namespace with attributes svd_rank, hops, svd_power_iters (optional)
        adj_low: PyTorch sparse tensor (COO) adjacency on CPU or GPU
        features: torch.Tensor of shape [N, D] on GPU
        device: target device (e.g., torch.device('cuda'))
    
    Returns:
        low_features: list of [N, D] tensors for each hop low-frequency part
        high_features: list of [N, D] tensors for each hop high-frequency part
    """
    t0 = perf_counter()
    
    A = adj_low.coalesce().to(device)            
    N = A.size(0)
    k = min(args.svd_rank, N)
    n_iter = getattr(args, "svd_power_iters", 2)

    Ω = torch.randn(N, k, device=device)
    Y = torch.sparse.mm(A, Ω)                  

    for _ in range(n_iter):
        Y = torch.sparse.mm(A, torch.sparse.mm(A.T, Y))

    Q, _   = torch.linalg.qr(Y)                  #
    AQ     = torch.sparse.mm(A, Q)              
    B      = Q.T @ AQ                           
    Û, S, V̂t = torch.linalg.svd(B)             

    U  = Q @ Û                                
    Vt = V̂t @ Q.T                               

    VtX = Vt @ features                          
    onesN = torch.ones(N, 1, device=device)
    Vt1   = Vt @ onesN                          

    low_features  = [features]                  
    high_features = [features]

    A2_dense = U @ (torch.diag(S ** 2) @ Vt)          
    A2 = A2_dense.to_sparse().coalesce()             
    adj_low_unnormalized = [adj_low, A2]              

    use_w = getattr(args, "normalization", "").lower() == "global_w"

    for hop in range(1, args.hops):
        S_k    = S ** hop
        S_km1  = S ** (hop - 1)
        S_diff = S_k - S_km1 - S                    

        Delta   = torch.diag(S_diff)             
        low_h   = U @ (Delta @ VtX)              

        if use_w:
            row_sum = U @ (Delta @ Vt1)          
            w       = torch.sigmoid(row_sum)     
            low_h_out   = w * low_h
            high_h_out  = (1.0 - w) * low_h - features

        else:
            low_h_out   = low_h
            high_h_out  = features - low_h_out

        low_features.append(low_h_out)
        high_features.append(high_h_out)

    del A, Ω, Y, Q, AQ, B, Û, V̂t, VtX, Vt1, onesN
    gc.collect(); torch.cuda.empty_cache()

    print(f"[distinct_hop_precompute_randomized] time: {perf_counter()-t0:.4f}s")
    return low_features, high_features, adj_low_unnormalized


def distinct_hop_precompute(args, adj_low, adj_high, I, features, device):

    t = perf_counter()

    current_A_EXP_low = adj_low
    low_channels = [I, (adj_low - I)]
    prev_A_EXP_low = adj_low
    
    # For hard masking: track cumulative sum of all previous hop matrices
    masking_mode = getattr(args, 'masking', 'adaptive')
    if masking_mode == 'hard':
        cumulative_mask = I + (adj_low - I)  # A^(0) + A^(1)

    for i in range(1, args.hops - 1):
        current_A_EXP_low = torch.spmm(current_A_EXP_low, adj_low)  
        
        if masking_mode == 'hard':
            # Hard masking: Mask(A^k, sum of all previous A^(j))
            # Set entries to 0 where cumulative_mask is non-zero
            current_A_EXP_low = current_A_EXP_low.coalesce()
            cumulative_mask = cumulative_mask.coalesce()
            
            # Create mask: keep only entries where cumulative_mask == 0
            indices = current_A_EXP_low.indices()
            values = current_A_EXP_low.values()
            
            # Check which entries in cumulative_mask are zero
            mask_indices = cumulative_mask.indices()
            mask_dict = set()
            for j in range(mask_indices.size(1)):
                mask_dict.add((mask_indices[0, j].item(), mask_indices[1, j].item()))
            
            # Keep only values where the position is NOT in cumulative_mask
            keep_mask = []
            for j in range(indices.size(1)):
                pos = (indices[0, j].item(), indices[1, j].item())
                keep_mask.append(pos not in mask_dict)
            
            keep_mask = torch.tensor(keep_mask, device=device)
            filtered_indices = indices[:, keep_mask]
            filtered_values = values[keep_mask]
            
            distinct_A_EXP_low = torch.sparse_coo_tensor(
                filtered_indices, filtered_values, 
                current_A_EXP_low.size(), device=device
            ).coalesce()
            
            # Update cumulative mask for next iteration
            cumulative_mask = (cumulative_mask + current_A_EXP_low).coalesce()
        else:
            # Adaptive masking: ReLU(A^k - A^(k-1))
            distinct_A_EXP_low = current_A_EXP_low - prev_A_EXP_low
            distinct_A_EXP_low = distinct_A_EXP_low.coalesce()
            
            # Apply ReLU: keep only positive values
            indices = distinct_A_EXP_low.indices()
            values = distinct_A_EXP_low.values()
            positive_mask = values > 0
            
            distinct_A_EXP_low = torch.sparse_coo_tensor(
                indices[:, positive_mask],
                values[positive_mask],
                distinct_A_EXP_low.size(),
                device=device
            ).coalesce()

        low_channels.append(distinct_A_EXP_low)
        prev_A_EXP_low = current_A_EXP_low

    del current_A_EXP_low, distinct_A_EXP_low, prev_A_EXP_low
    gc.collect()
    torch.cuda.empty_cache()


    low_channels_dense = []

    for k, channel in enumerate(low_channels):           
        norm = args.normalization.lower()
        channel = channel.coalesce()
        if norm == "sparse_row_sum":                     
            row_deg = torch.sparse.sum(channel, dim=1).to_dense() + 1e-9 
            idx_i, idx_j = channel.indices()
            vals = channel.values() / row_deg[idx_i]
            channel_norm = torch.sparse_coo_tensor(
                channel.indices(), vals, channel.size(), device=channel.device
            )
            ch_dense = channel_norm.to_dense()

        elif norm == "sparse_row_sum_gain":              
            row_deg = torch.sparse.sum(channel, dim=1).to_dense() + 1e-9
            alpha_k = row_deg.mean()                      
            idx_i = channel.indices()[0]
            vals = alpha_k * channel.values() / row_deg[idx_i]
            channel_norm = torch.sparse_coo_tensor(
                channel.indices(), vals, channel.size(), device=channel.device
            )
            ch_dense = channel_norm.to_dense()
        elif norm == "global_gain":
            mean_row = torch.sparse.sum(channel, dim=1).to_dense().mean() + 1e-9
            target = 1.0                                
            alpha_k = target / mean_row                 
            channel_norm = torch.sparse_coo_tensor(
                channel.indices(),
                alpha_k * channel.values(),             
                channel.size(),
                device=channel.device,
            ).coalesce()                               
            ch_dense = channel_norm.to_dense()

        elif norm == "global_scale":                      
            target_mean = 1.0                             
            s = torch.sparse.sum(channel).item() / channel.size(0) + 1e-9
            gamma_k = target_mean / s
            channel_norm = torch.sparse_coo_tensor(
                channel.indices(), gamma_k * channel.values(),
                channel.size(), device=channel.device
            )
            ch_dense = channel_norm.to_dense()

        elif norm == "log":                              
            channel_norm = torch.sparse_coo_tensor(
                channel.indices(), torch.log1p(channel.values()),
                channel.size(), device=channel.device
            )
            ch_dense = channel_norm.to_dense()

        elif norm == "softmax":
            ch_dense = channel.to_dense()
            ch_dense = torch.nn.functional.softmax(ch_dense, dim=1)

        elif norm == "row_sum":
            ch_dense = channel.to_dense()
            row_sums = ch_dense.sum(dim=1, keepdim=True) + 1e-9
            ch_dense = ch_dense / row_sums

        else:
            ch_dense = channel.to_dense()

        low_channels_dense.append(ch_dense)
        del ch_dense

    I_dense = I.to_dense()
    high_channels_dense = [I_dense]
    for ch in low_channels_dense[1:]:
        high_channels_dense.append(I_dense - ch)

    precompute_time = perf_counter() - t
    print(f"DistinctHop Precompute time: {precompute_time:.4f} seconds")

    return low_channels_dense, high_channels_dense


def train_prep(logger, args):
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    split_idx_lst = None

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  
    os.environ['PYTHONHASHSEED'] = str(args.seed) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)   
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        "svd_rank": args.svd_rank,
        "top": args.top,
        "comm_size": args.comm_size,
        "approach": args.approach,
        "masking": getattr(args, 'masking', 'adaptive')
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
    adj_low_raw, features, labels = load_full_data(args.dataset_name)
    nnodes = labels.shape[0]
    I = sp.eye(nnodes)

    if args.structure_info:
        adj_low = normalize_adj(adj_low_raw)

        adj_high = (I - adj_low)

    else:
        adj_low = normalize_adj(adj_low_raw)

        adj_high = (I - adj_low)
        adj_low_unnormalized = None

    if (args.model == "acmsgc") and (args.hops > 1):
        A_EXP = adj_low.to_dense()
        for _ in range(args.hops - 1):
            A_EXP = torch.mm(A_EXP, adj_low.to_dense())
        adj_low = A_EXP.to_sparse()
        del A_EXP
        adj_low = adj_low.to(device).to_sparse()


    if (args.model == "adaptcs") and (args.hops > 1):     
        t = perf_counter()

        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).float().to(device)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).float().to(device)
        I = sparse_mx_to_torch_sparse_tensor(I).float().to(device)

        if args.approach == 'distinct_hop':
            t = perf_counter()
            adj_low_unnormalized = [adj_low]
            adj_low, adj_high = distinct_hop_precompute(args, adj_low, adj_high, I, features, device)
            print(f"distinct_hop_precompute time: {perf_counter()-t:.4f} seconds")
        elif args.approach == 'distinct_hop_svds_low':
            t = perf_counter()
            adj_low, adj_high, adj_low_unnormalized = distinct_hop_precompute_svds_low(args, adj_low, adj_high, I, features, device)
            print(f"distinct_hop_precompute time: {perf_counter()-t:.4f} seconds")
            # raise Exception
        elif args.approach == 'distinct_hop_svds_rand':
            t = perf_counter()
            adj_low, adj_high, adj_low_unnormalized = distinct_hop_precompute_randomized_svds(args, adj_low, adj_high, I, features, device)
            print(f"distinct_hop_precompute time: {perf_counter()-t:.4f} seconds")
            # raise Exception

    
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




def cs_eval_metrics(communities, query_nodes, labels):
    true = [1] * (len([element for sublist in communities for element in sublist]) - len(query_nodes))
    pred = []
    for community , query in zip(communities, query_nodes):
        pred.append(np.equal(labels[torch.LongTensor(community[1:])].tolist(), int(labels[query])) * 1)

    pred = [element for sublist in pred for element in sublist]
    jaccard = round(jaccard_score(true, pred, average='binary'), 4)
    nmi = round(normalized_mutual_info_score(true, pred, average_method='arithmetic'), 4)
    f1 = round(f1_score(true, pred, average='binary'), 4)
    return f1, jaccard, nmi


def centroid_distance(community, features, prev_centroid=None, prev_node=None, new_node=None, community_size = 20):
    features_c = features[torch.LongTensor(community)]
    
    if prev_centroid is not None and prev_node is not None and new_node is not None:
        prev_centroid_1d = prev_centroid.squeeze(0) if prev_centroid.dim() == 2 else prev_centroid
        updated_centroid = prev_centroid_1d - (1 / community_size) * (features[prev_node] - features[new_node])
    else:
        updated_centroid = torch.sum(features_c, dim=0) / community_size

    updated_centroid = updated_centroid.unsqueeze(0)
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(features_c, updated_centroid.expand_as(features_c))
    return similarity, updated_centroid  


def sub_cs(features, query_nodes, community_size, early_stop):
    communities = []
    cs_start = perf_counter()

    for query in query_nodes:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_simi = cos(features, features[query].reshape(1, -1)).squeeze()
        topk_prob, topk_idx = torch.topk(cos_simi, community_size * 2)
        
        community = topk_idx[:community_size].tolist()
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


def subgraph(adjacency_matrix, query_nodes, subgraph_size, label, train_ratio):
        
    train_list = []
    test_list = []
    train_size = int(subgraph_size * train_ratio /2)
    for query in query_nodes:
        allnodes = [i for i in range(adjacency_matrix.shape[0])]
        sub_pos = []
        sub_neg = []
        for node in allnodes:
            if torch.equal(label[node], label[query]):
                sub_pos.append(node)
            else:
                sub_neg.append(node)
        if len(sub_pos)>=train_size:
            train_pos = random.sample(sub_pos, train_size) 
            train_neg = random.sample(sub_neg, train_size)
        else: 
            train_pos = random.sample(sub_pos, train_size-1) 
            train_neg = random.sample(sub_neg, train_size+1)

        sub_train = train_pos + train_neg
        sub_test = list(set(allnodes).difference(sub_train))

        train_list.append(sub_train)
        test_list.append(sub_test)
    train_list = torch.IntTensor(train_list).cuda()
    test_list = torch.IntTensor(test_list).cuda()

    return train_list, test_list #cs_time


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

    base_score = 0.0
    penalty_sum = 0.0
    bonus_sum = 0.0
    
    for i in range(len(candidate)):
        for j in range(i+1, len(candidate)):
            u = candidate[i]
            v = candidate[j]
            
            # Retrieve S[u,v]
            s_val = S[u, v]
            if isinstance(s_val, torch.Tensor):
                s_val = s_val.item()
            
            base_score += s_val
            
            if adj[u, v] != 0:
                if s_val < threshold:
                    penalty_sum += (threshold - s_val)
            else:
                if adj_2[u, v] > 0 and s_val > threshold:
                    bonus_sum += adj_2[u, v]
    
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
   
    cs_start = perf_counter()
    communities = []
    N = S.shape[0]
    
    for q in query_nodes:
        sim_q = S[q,:]  
        top_vals, top_indices = torch.topk(sim_q, community_size * top_factor)
        candidate = top_indices.tolist()
        
        if q not in candidate:
            candidate[-1] = q  
        current_obj = community_objective_extend(candidate, S, adj, adj_2,
                                                threshold, lambda_pen, lambda_2hop)
        
        while len(candidate) > community_size:
            best_obj = -1e15
            best_node = None
            for node in candidate:
                new_cand = [x for x in candidate if x != node]
                test_obj = community_objective_extend(new_cand, S, adj, adj_2,
                                                      threshold, lambda_pen, lambda_2hop)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_node = node
            if best_obj > current_obj:
                candidate.remove(best_node)
                current_obj = best_obj
            else:
                break
        
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

    t0 = perf_counter()
    communities = []
    
    use_torch = isinstance(S, torch.Tensor)
    
    N = S.shape[0]
    
    def get_val(matrix, u, v):
        val = matrix[u, v]
        if isinstance(val, torch.Tensor):
            return val.item()
        return val
    
    for q in query_nodes:
        if use_torch:
            sim_q = S[q]  
            top_vals, top_idx = torch.topk(sim_q, community_size * top_factor)
            candidate = top_idx.tolist()
        else:
            sim_q = S[q]
            candidate = np.argsort(-sim_q)[:community_size * top_factor].tolist()
        
        if q not in candidate:
            candidate[-1] = q
        
        cand_set = set(candidate)
        
        base_sum = 0.0
        penalty_sum = 0.0
        bonus_sum = 0.0
        
        base_contrib = {}
        penalty_contrib = {}
        bonus_contrib = {}
        
        for u in candidate:
            bc = 0.0
            pc = 0.0
            bnc = 0.0
            for v in candidate:
                if v == u:
                    continue
                s_uv = get_val(S, u, v)
                
                bc += s_uv
                
                if get_val(adj, u, v) != 0:
                    if s_uv < threshold:
                        pc += (threshold - s_uv)
                else:
                    if get_val(adj_2, u, v) > 0 and s_uv > threshold:
                        bnc += get_val(adj_2, u, v)
            
            base_contrib[u] = bc
            penalty_contrib[u] = pc
            bonus_contrib[u] = bnc
            
            base_sum += bc
            penalty_sum += pc
            bonus_sum += bnc
        
        base_sum    *= 0.5
        penalty_sum *= 0.5
        bonus_sum   *= 0.5
        
        current_obj = community_objective_from_sums(base_sum, penalty_sum, bonus_sum,
                                                    lambda_pen, lambda_2hop)
        
        while len(candidate) > community_size:
            best_obj = -1e15
            best_node = None
            
            for node in candidate:
                new_base = base_sum - (base_contrib[node] * 0.5)
                new_pen = penalty_sum - (penalty_contrib[node] * 0.5)
                new_bon = bonus_sum - (bonus_contrib[node] * 0.5)
                
                test_obj = community_objective_from_sums(new_base, new_pen, new_bon,
                                                         lambda_pen, lambda_2hop)
                if test_obj > best_obj:
                    best_obj = test_obj
                    best_node = node
            
            if best_obj > current_obj and best_node is not None:
                r = best_node
                candidate.remove(r)
                cand_set.remove(r)
                
                base_sum  -= (base_contrib[r] * 0.5)
                penalty_sum -= (penalty_contrib[r] * 0.5)
                bonus_sum   -= (bonus_contrib[r] * 0.5)
                
                for other in candidate:
                    s_ov = get_val(S, other, r)
                    
                    base_contrib[other] -= s_ov
                    
                    if get_val(adj, other, r) != 0:
                        if s_ov < threshold:
                            pen_val = (threshold - s_ov)
                            penalty_contrib[other] -= pen_val
                    else:
                        if get_val(adj_2, other, r) > 0 and s_ov > threshold:
                            bon_val = get_val(adj_2, other, r)
                            bonus_contrib[other] -= bon_val
                
                del base_contrib[r]
                del penalty_contrib[r]
                del bonus_contrib[r]
                
                current_obj = best_obj
            else:
                break  
        
        if len(candidate) > community_size:
            candidate = candidate[:community_size]
        
        communities.append(candidate)
    
    total_time = perf_counter() - t0
    avg_time = total_time / len(query_nodes) if len(query_nodes) else 0.0
    return communities, avg_time


def _has_two_hop(u: int, v: int, rowptr, col):
    a0, a1 = rowptr[u].item(), rowptr[u + 1].item()
    b0, b1 = rowptr[v].item(), rowptr[v + 1].item()
    i, j = a0, b0
    while i < a1 and j < b1:
        nu, nv = int(col[i]), int(col[j])
        if nu == nv:
            return True
        if nu < nv:
            i += 1
        else:
            j += 1
    return False


def signed_cs_fast_light(
    norm_feats: torch.Tensor,    
    query_nodes,
    adj_sparse: torch.Tensor,     
    tau: float = 0.2,
    lambda_pen: float = 1.0,
    lambda_2hop: float = 0.5,
    k: int = 10,
    top_factor: int = 2,
):
    N, D = norm_feats.shape
    comms, t0 = [], perf_counter()

    csr = adj_sparse.coalesce()
    if hasattr(torch_sparse.SparseTensor, "from_torch_sparse_coo"):
        st = torch_sparse.SparseTensor.from_torch_sparse_coo(csr)
    else:
        st = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(csr)
    rowptr, col_idx, _ = st.csr()
    rowptr, col_idx = rowptr.cpu(), col_idx.cpu()

    def top_m(q: int, m: int):
        sim = torch.mv(norm_feats, norm_feats[q])       
        _, idx = sim.topk(m + 1)
        return idx[idx != q][:m].tolist()

    def cos(u, v):
        return float((norm_feats[u] * norm_feats[v]).sum())

    def one_hop(u, v):
        s, e = rowptr[u].item(), rowptr[u + 1].item()
        return (col_idx[s:e] == v).any().item()

    for q in query_nodes:
        cand = top_m(int(q), k * top_factor)
        if int(q) not in cand:
            cand[-1] = int(q)

        k_now = len(cand)
        X = norm_feats[cand]                      
        C = X.mean(dim=0)                          
        normC2 = float((C * C).sum())
        base_sum = k_now * normC2 - k_now - k_now * (k_now - 1) * tau

        base_c = {
            u: k_now * float((C * norm_feats[u]).sum()) - 1 - (k_now - 1) * tau
            for u in cand
        }

        pen_c = {}
        bon_c = {}
        pen_sum = bon_sum = 0.0
        for u in cand:
            pc = bc = 0.0
            for v in cand:
                if u == v:
                    continue
                s_uv = cos(u, v) - tau
                if one_hop(u, v):
                    if s_uv < 0:
                        pc += -s_uv
                elif _has_two_hop(u, v, rowptr, col_idx) and s_uv > 0:
                    bc += 1.0
            pen_c[u], bon_c[u] = pc, bc
            pen_sum += pc
            bon_sum += bc
        pen_sum *= .5
        bon_sum *= .5

        best_obj = (base_sum + lambda_2hop * bon_sum - lambda_pen * pen_sum)

        while len(cand) > k:
            drop, gain = None, -float('inf')
            for u in cand:
                nb  = base_sum - 0.5 * base_c[u]
                npen = pen_sum - 0.5 * pen_c[u]
                nbon = bon_sum - 0.5 * bon_c[u]
                obj = nb + lambda_2hop * nbon - lambda_pen * npen
                if obj > gain:
                    drop, gain = u, obj

            if gain <= best_obj:
                break 

            cand.remove(drop)
            k_now -= 1

            dot_cd = float((C * norm_feats[drop]).sum())
            normC2 = (k_now + 1) / k_now ** 2 * normC2 - 2 * dot_cd / k_now ** 2 + 1 / k_now ** 2
            C = (C * (k_now + 1) - norm_feats[drop]) / k_now
            base_sum = k_now * normC2 - k_now - k_now * (k_now - 1) * tau

            for v in cand:
                if v == drop:
                    continue
                base_c[v] = k_now * float((C * norm_feats[v]).sum()) - 1 - (k_now - 1) * tau
                s_vd = cos(v, drop) - tau
                if one_hop(v, drop):
                    if s_vd < 0:
                        pen_c[v] += s_vd  
                elif _has_two_hop(v, drop, rowptr, col_idx) and s_vd > 0:
                    bon_c[v] -= 1.0
            base_c.pop(drop, None)
            pen_c.pop(drop, None)
            bon_c.pop(drop, None)

            pen_sum = sum(pen_c.values()) * 0.5
            bon_sum = sum(bon_c.values()) * 0.5
            best_obj = gain

        comms.append(cand[:k])

    avg_t = (perf_counter() - t0) / max(len(query_nodes), 1.)
    return comms, avg_t

def adaptive_score_function(u, v, sim, is_edge, h, tau):
    s_uv = sim - tau
    
    base_score = s_uv
    
    if is_edge:
        consistency = h * max(0, s_uv) + (1-h) * max(0, -s_uv)
    else:
        if h > 0.7:  
            consistency = max(0, s_uv)
        elif h < 0.3:  
            consistency = 1 - abs(sim - 0.5) * 2
        else:  
            consistency = 0.5 * (1 + s_uv) if sim > tau else 0
    
    return base_score + consistency

def adaptive_cs_fast(
    norm_feats: torch.Tensor,    
    query_nodes,
    adj_sparse: torch.Tensor,     
    tau: float = 0.2,
    lambda_consistency: float = 1.0,
    k: int = 10,
    top_factor: int = 2,
    h: float = None,              
):
    """自适应同配/异配图的快速社区搜索"""
    N, D = norm_feats.shape
    comms, t0 = [], perf_counter()
    
    csr = adj_sparse.coalesce()
    st = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(csr)
    rowptr, col_idx, _ = st.csr()
    rowptr, col_idx = rowptr.cpu(), col_idx.cpu()
    
    if h is None:
        h = compute_homophily_1(norm_feats, adj_sparse)
    
    def top_m(q, m):
        sim = torch.mv(norm_feats, norm_feats[q])
        _, idx = sim.topk(m + 1)
        return idx[idx != q][:m].tolist()
    
    def get_neighbors(u):
        s, e = rowptr[u].item(), rowptr[u+1].item()
        return set(col_idx[s:e].tolist())
    
    for q in query_nodes:
        cand = top_m(int(q), k * top_factor)
        if int(q) not in cand:
            cand[-1] = int(q)
        
        cand_sim = {}
        cand_neighbors = {}
        for u in cand:
            cand_neighbors[u] = get_neighbors(u)
            for v in cand:
                if u < v:
                    sim_uv = float((norm_feats[u] * norm_feats[v]).sum())
                    cand_sim[(u, v)] = sim_uv
        
        total_score = 0.0
        node_scores = {u: 0.0 for u in cand}
        
        for (u, v), sim in cand_sim.items():
            is_edge = v in cand_neighbors[u]
            score_uv = adaptive_score_function(u, v, sim, is_edge, h, tau)
            
            total_score += score_uv
            node_scores[u] += score_uv
            node_scores[v] += score_uv
        
        while len(cand) > k:
            min_score, to_remove = float('inf'), None
            for u in cand:
                if node_scores[u] < min_score:
                    min_score, to_remove = node_scores[u], u
            
            cand.remove(to_remove)
            total_score -= node_scores.pop(to_remove)
            
            for u in cand:
                if (u, to_remove) in cand_sim or (to_remove, u) in cand_sim:
                    pair = (min(u, to_remove), max(u, to_remove))
                    sim_uv = cand_sim[pair]
                    is_edge = to_remove in cand_neighbors[u]
                    
                    old_score_uv = adaptive_score_function(u, to_remove, sim_uv, is_edge, h, tau)
                    node_scores[u] -= old_score_uv
                    total_score -= old_score_uv
                    
                    del cand_sim[pair]
        
        comms.append(cand[:k])
    
    avg_t = (perf_counter() - t0) / max(len(query_nodes), 1.0)
    return comms, avg_t

def compute_homophily_1(features, adj_sparse, sample_size=1000):
    """计算图的同配系数 (0-1)"""
    indices = adj_sparse.indices()
    num_edges = indices.shape[1]
    
    sample_indices = torch.randint(0, num_edges, (min(sample_size, num_edges),))
    sampled_edges = indices[:, sample_indices]
    
    src_nodes = sampled_edges[0]
    dst_nodes = sampled_edges[1]
    src_feats = features[src_nodes]
    dst_feats = features[dst_nodes]
    
    similarities = torch.sum(src_feats * dst_feats, dim=1)
    avg_similarity = torch.mean(similarities).item()
    return (avg_similarity + 1) / 2





def to_sparse_tensor(mat) -> SparseTensor:
    if isinstance(mat, SparseTensor):
        return mat
    if not mat.is_sparse:
        mat = mat.to_sparse()
    return SparseTensor.from_torch_sparse_coo_tensor(mat.coalesce())


def compute_homophily(adj, X, num_samples: int = 2000) -> float:
    adj = to_sparse_tensor(adj)
    src, dst, _ = adj.coo()
    m = src.numel()
    samp = torch.randint(0, m, (min(num_samples, m),), device=src.device)
    cos = F.cosine_similarity(X[src[samp]], X[dst[samp]], dim=1)   
    return float((cos.mean() + 1) * 0.5)                          


def adaptive_cs_top2k(query_nodes: List[int],
                      adj_in,
                      X_norm: torch.Tensor,
                      k: int = 10,
                      tau: float = 0.2,
                      bonus: float = 1.0,
                      penalty: float = 1.0,
                      top_factor: int = 2
                      ):

    adj = to_sparse_tensor(adj_in).cpu()
    rowptr, col, _ = adj.csr()
    rowptr, col = rowptr.numpy(), col.numpy()

    def is_neighbor(u: int, v: int) -> bool:
        u = int(u); v = int(v)  
        beg, end = rowptr[u], rowptr[u+1]
        return v in col[beg:end]

    h = compute_homophily(adj, X_norm)          
    w_edge = (1.0 - tau) * (  h * bonus       )   
    w_pen  = (1.0 - tau) * -((1-h) * penalty )   

    communities, t0 = [], perf_counter()
    X_cpu = X_norm.cpu()

    for q in query_nodes:
        q = int(q)
        sims = torch.mv(X_cpu, X_cpu[q])                     
        top_idx = sims.topk(top_factor * k + 1).indices.tolist()
        C = {i for i in top_idx if i != q} | {q}

        scores = []
        for u in C:
            sim_part = sims[u].item()                         
            if is_neighbor(q, u):
                edge_part = w_edge if h >= 0.5 else w_pen    
            else:
                edge_part = 0.0
            scores.append( (sim_part + edge_part, u) )

        scores.sort(reverse=True)            
        community = [u for _, u in scores[:k]]
        communities.append(community)

    avg_time = (perf_counter() - t0) / max(len(query_nodes), 1)
    return communities, avg_time






def precompute_neighbor_lists(adj, adj_2):
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
    device = emb.device
    N = emb.size(0)
    
    norms = emb.norm(dim=1, keepdim=True)
    normed_emb = emb / (norms + 1e-12)
    sim_matrix = normed_emb @ normed_emb.t() 
    
    S = sim_matrix - threshold
    return S

def build_positive_graph_torch(S):
    row_idx, col_idx = torch.where(S >= 0)
    
    N = S.size(0)
    pos_graph = [[] for _ in range(N)]
    
    for i in range(len(row_idx)):
        u = row_idx[i].item()
        v = col_idx[i].item()
        if u != v:
            pos_graph[u].append(v)
    
    return pos_graph

def bfs_with_teleport(pos_graph, query, sim_q, k):
   
    from collections import deque
    
    visited = set([query])
    queue = deque([query])
    N = len(pos_graph)
    
    while True:
        while queue and len(visited) < k:
            u = queue.popleft()
            for v in pos_graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        if len(visited) >= k:
            break  # done

        if not queue and len(visited) < k:
            candidates = []
            for node_idx in range(N):
                if node_idx not in visited:
                    sim_val = float(sim_q[node_idx])
                    candidates.append((sim_val, node_idx))
            if not candidates:
                break 
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_node = candidates[0][1]
            visited.add(best_node)
            queue.append(best_node)

        if not queue:
            pass
    
    return list(visited)


def bfs_teleport(emb, query_nodes, k, threshold=0.2):
   
    cs_start = perf_counter()

    S = build_signed_adjacency(emb, threshold=threshold)
    
    pos_graph = build_positive_graph_torch(S)
    
    communities = []
    
    with torch.no_grad():
        norms = emb.norm(dim=1, keepdim=True)
        normed_emb = emb / (norms + 1e-12)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    for q in query_nodes:
        cos_simi = cos(normed_emb, normed_emb[q].reshape(1, -1)).squeeze()
        comm = bfs_with_teleport(pos_graph, q, cos_simi, k)
        
        if q in comm:
            comm.remove(q)
            comm.insert(0, q)
        communities.append(comm)
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    
    return communities, cs_time



def coo_to_csr(adj_coo: torch.Tensor):

    adj = adj_coo.coalesce()
    row, col = adj.indices()
    N = adj.size(0)
    E = row.numel()

    if not torch.all(row[1:] >= row[:-1]):
        order = torch.argsort(row * N + col)
        row, col = row[order], col[order]

    rowptr = torch.zeros(N + 1, dtype=torch.long, device=row.device)
    ones = torch.ones(E, dtype=torch.long, device=row.device)
    rowptr.index_add_(0, row, ones)
    rowptr = torch.cumsum(rowptr, 0)          
    return rowptr, col


@torch.no_grad()
def build_positive_graph_rowwise(adj_sparse: torch.Tensor,
                                 norm_feats: torch.Tensor,
                                 threshold: float) -> List[List[int]]:
    """
    Parameters
    ----------
    adj_sparse : torch.sparse_coo_tensor  (N×N), 无向或有向皆可
    norm_feats : 已经 L2 归一化的嵌入 (N, F)
    threshold  : 余弦阈值， >=threshold 的边被视为正边

    Returns
    -------
    pos_graph : Python list-of-list，len=N
                pos_graph[u] = [v1, v2, ...] (u→v 方向正边)
    """
    rowptr, col_idx = coo_to_csr(adj_sparse)  
    N, F = norm_feats.shape
    pos_graph = [[] for _ in range(N)]

    for u in range(N):
        beg, end = rowptr[u].item(), rowptr[u + 1].item()
        if beg == end:
            continue
        nbrs = col_idx[beg:end]                
        sim = (norm_feats[nbrs] * norm_feats[u]).sum(dim=1)
        keep = sim >= threshold
        pos_graph[u].extend(nbrs[keep].tolist())

    return pos_graph


def _bfs_with_teleport(pos_graph: List[List[int]],
                       query: int,
                       sim_q: torch.Tensor,
                       k: int) -> List[int]:
    visited = {query}
    q = deque([query])
    N = len(pos_graph)
    sim_arr = sim_q.detach().cpu().tolist()   

    while len(visited) < k:
        while q and len(visited) < k:
            u = q.popleft()
            for v in pos_graph[u]:
                if v not in visited:
                    visited.add(v)
                    if len(visited) == k:
                        break
                    q.append(v)

        if len(visited) == k:
            break

        if not q:
            best = max(
                (idx for idx in range(N) if idx not in visited),
                key=lambda idx: sim_arr[idx],
                default=None)
            if best is None:
                break
            visited.add(best)
            q.append(best)

    return list(visited)


def bfs_teleport_sparse(adj_sparse: torch.Tensor,
                        emb: torch.Tensor,
                        query_nodes: Sequence[int],
                        k: int,
                        threshold: float = 0.2):
   
    t0 = perf_counter()

    norm_feats = torch.nn.functional.normalize(emb, p=2, dim=1)

    pos_graph = build_positive_graph_rowwise(adj_sparse, norm_feats, threshold)

    communities = []
    for q in query_nodes:
        sim_q = torch.mv(norm_feats, norm_feats[q])   # (N,)
        comm = _bfs_with_teleport(pos_graph, int(q), sim_q, k)
        communities.append(comm)

    avg_time = (perf_counter() - t0) / max(len(query_nodes), 1)
    return communities, avg_time



def bfs_single(rowptr, col_idx, sim_vec, query, k, thresh):
    N = rowptr.numel() - 1
    visited = torch.zeros(N, dtype=torch.bool, device=sim_vec.device)
    q = deque([query])
    comm = []
    visited[query] = True

    while q:
        v = q[0]
        if len(comm) < k:
            comm.append(v); q.popleft()
        else:
            idx_tensor = torch.as_tensor(comm, dtype=torch.long, device=sim_vec.device)
            probs      = sim_vec[idx_tensor]     
            min_p, idx_min = torch.min(probs, dim=0)
            if comm[idx_min] == query:
                break
            if sim_vec[v] > min_p and sim_vec[v] >= thresh:
                comm[idx_min] = v
            q.popleft()

        beg, end = rowptr[v].item(), rowptr[v + 1].item()
        for nb in col_idx[beg:end]:
            nb = nb.item()
            if not visited[nb]:
                q.append(nb); visited[nb] = True

    if query in comm:
        comm.remove(query)
    comm.insert(0, query)
    return comm


def bfs(adj,              
                 emb,           
                 query_nodes,    
                 k, thresh=0.5):

    ei = adj.coalesce().indices()
    csr = torch_sparse.SparseTensor(row=ei[0], col=ei[1],
                       sparse_sizes=adj.shape).coalesce()
    rowptr, col_idx, _ = csr.csr()

    emb_n = torch.nn.functional.normalize(emb, p=2, dim=1)

    communities = []
    t0 = perf_counter()

    for q in query_nodes:
        sim_vec = emb_n @ emb_n[q]          
        comm = bfs_single(rowptr, col_idx, sim_vec, q, k, thresh)
        communities.append(comm)

    cs_time = (perf_counter() - t0) / len(query_nodes)
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
        cos_simi = cos(features, features[query].reshape(1, -1)).squeeze()
        _, topk_idx = torch.topk(cos_simi, community_size)
        community = topk_idx.tolist()
        communities.append(community)
    
    cs_time = (perf_counter() - cs_start) / len(query_nodes)
    return communities, cs_time


def community_search(adj, query_nodes, emb, community_size, labels, method='sub_cs'):
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
   
    if args.model in ['adaptcs']:

        if method == 'bfs_teleport':
            communities, cs_time = bfs_teleport_sparse(adj, emb, query_nodes, community_size, args.threshold)
        
            
        elif method == 'adaptive_cs':
            # adj = adj.coalesce()
            norm_feats = torch.nn.functional.normalize(emb, p=2, dim=1)
            communities, cs_time = adaptive_cs_top2k(query_nodes, 
                                                     adj, norm_feats,
                                                     k=community_size, 
                                                     tau=args.threshold, 
                                                     top_factor = args.top)

        elif method == 'bfs':
            communities, cs_time = bfs(emb, query_nodes, adj, community_size, args.threshold)
        else:
            raise ValueError("Invalid community search method")
        
        cs_f1, cs_jaccard, cs_nmi = cs_eval_metrics(communities, query_nodes, labels)
        
    else:
        communities, cs_time = bfs(adj, emb, query_nodes, community_size, 0.9)
        cs_f1, cs_jaccard, cs_nmi = cs_eval_metrics(communities, query_nodes, labels)
    
    return cs_f1, cs_jaccard, cs_nmi, cs_time