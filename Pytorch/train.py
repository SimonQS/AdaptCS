from __future__ import division, print_function
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from arg_parser import arg_parser
from models.models import GCN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils import (
    evaluate,
    eval_acc,
    data_split,
    random_disassortative_splits,
    train_model,
    train_prep,
    community_search,
)

args = arg_parser()
from logger import ACMPythorchLogger   
logger = ACMPythorchLogger(csv_path=args.batch_csv)



(device,
model_info,
run_info,
adj_high,
adj_low,
adj_low_unnormalized,
features,
labels,
split_idx_lst,
) = train_prep(logger, args)

criterion = nn.NLLLoss()
eval_func = eval_acc

t_total = time.time()
epoch_total = 0
result_splits = np.zeros(args.num_splits)

run_info.update({"lr": args.lr, "weight_decay": args.weight_decay, "dropout": args.dropout})

best_emb = None          
for split_id in range(args.num_splits):
    run_info["split"] = split_id

    if args.fixed_splits == 0:
        idx_train, idx_val, idx_test = random_disassortative_splits(labels, labels.max() + 1)
    else:
        idx_train, idx_val, idx_test = data_split(split_id, args.dataset_name)

    queries = torch.nonzero(idx_train, as_tuple=True)[0].tolist()


    model = GCN(
        nfeat=features.size(1),
        nhid=args.hidden,
        nclass=int(labels.max()) + 1,
        nlayers=args.layers,
        nnodes=features.size(0),
        dropout=args.dropout,
        model_type=args.model,
        structure_info=args.structure_info,
        variant=args.variant,
        normalization=args.normalization,
        resnet=args.resnet,
        hops=args.hops,
        init_layers_X=args.link_init_layers_X,
        query=queries,
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        idx_train, idx_val, idx_test = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()
        model.cuda()

    best_val_loss = float("inf")
    best_epoch    = -1                       
    patience_cnt  = 0                        

    
    for epoch in range(args.epochs):
        train_model(
            model, optimizer,
            adj_low, adj_high, adj_low_unnormalized[0],
            features, labels, idx_train,
            criterion, dataset_name=args.dataset_name,
        )

        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output, emb, att = model(features, adj_low, adj_high, adj_low_unnormalized[0])
                output   = F.log_softmax(output, dim=1)
                val_loss = criterion(output[idx_val], labels[idx_val]).item()
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_epoch    = epoch
                    patience_cnt  = 0
                    curr_res = evaluate(output, labels, idx_test, eval_func)
                    best_emb = emb.detach()
                    best_att = att
                else:
                    patience_cnt += 1

                if args.early_stopping > 0 and patience_cnt >= args.early_stopping:
                    run_info[f"split{split_id}_early_stop_epoch"] = epoch
                    run_info[f"split{split_id}_early_stop_val"]   = float(best_val_loss)
                    break

    epoch_total += epoch
    result_splits[split_id] = curr_res
    del model, optimizer
    torch.cuda.empty_cache()
   

total_time = time.time() - t_total
run_info.update(
    {
        "runtime_average": total_time / args.num_splits,
        "epoch_average": total_time / epoch_total * 1000,
        "result": float(np.mean(result_splits)),
        "std":    float(np.std(result_splits)),
    }
)



print(f"Total time for training: {total_time:.4f}s")

with torch.no_grad():
    if (args.model == 'adaptcs' and 
        (args.approach == 'distinct_hop_svds_low' or 
        args.approach == 'distinct_hop_svds_rand')):
        query_nodes = torch.randperm(labels[idx_test].size(0), device=device)[:50]
        cs_f1, cs_jaccard, cs_nmi, cs_time = community_search(
            adj_low_unnormalized[0], query_nodes, best_emb, args.comm_size, labels,
            method=args.online_cs,
        )
    elif (args.model == 'adaptcs' and args.approach == 'distinct_hop'):
        query_nodes = torch.randperm(labels[idx_test].size(0), device=device)[:50]
        cs_f1, cs_jaccard, cs_nmi, cs_time = community_search(
            adj_low[1], query_nodes, best_emb, args.comm_size, labels,
            method=args.online_cs,
        )
    else:
        query_nodes = torch.randperm(labels[idx_test].size(0), device=device)[:50]
        cs_f1, cs_jaccard, cs_nmi, cs_time = community_search(
            adj_low[0], query_nodes, best_emb, args.comm_size, labels,
            method=args.online_cs,
        )
run_info.update(
    {
        "cs_time": cs_time,
        "cs_nmi":  cs_nmi,
        "cs_jaccard": cs_jaccard,
        "cs_f1": cs_f1,
        "cs_nmi_std": 0.0,
        "cs_jaccard_std": 0.0,
        "cs_f1_std": 0.0,
    }
)

logger.log_time(f"{total_time:.4f}s")
logger.log_run(model_info, run_info)


def plot_tsne_to_file(embeddings: torch.Tensor,
                      labels:     torch.Tensor,
                      save_name:  str = "tsne.png",
                      perplexity: int  = 30,
                      seed:       int  = 42,
                      figsize:    tuple = (6, 6),
                      dpi:        int  = 120):
    X = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
        random_state=seed,
    )
    Z = tsne.fit_transform(X)       

    num_cls = len(np.unique(y))
    cmap = plt.get_cmap("tab10", num_cls)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(Z[:, 0], Z[:, 1],
                c=y, s=8, cmap=cmap,
                alpha=0.8, linewidths=0)
    plt.xticks([]); plt.yticks([])
    plt.title(f"t-SNE  (perplexity={perplexity})", fontsize=12)

    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", save_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[t-SNE] image saved to {out_path}")

def plot_hop_attention(att: torch.Tensor,
                       save_path: str = "plots/hop_att_curve.png",
                       figsize=(5, 3), dpi=120):

    T = att.size(0)
    att_2d = att.squeeze(-1)
    gamma = att_2d.mean(dim=1).cpu().numpy()      # shape (T,)
    print(gamma)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(range(T), gamma, marker="o")
    plt.xlabel("k (hop)")
    plt.ylabel(r"$\gamma_k$")
    plt.title("Hop-wise Attention Weights")
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] hop attention image saved to {save_path}")


def plot_hop_attention_density(att: torch.Tensor,
                            save_path: str = "plots/hop_att_density.png",
                            figsize=(4, 3),
                            dpi=150):
    """
    att : Tensor [T, N, 1] or [T, N]
    """
    T = att.size(0)
    att_2d = att.squeeze(-1).cpu().numpy()  # [T, N]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    palette = ["royalblue", "darkorange", "limegreen", "crimson", "gray"]
    labels = [r"$\alpha_0$", r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_4$"]

    for k in range(T):
        sns.kdeplot(
            att_2d[k], fill=True, linewidth=1.5,
            color=palette[k % len(palette)],
            label=labels[k] if k < len(labels) else f"hop{k}"
        )

    plt.xlabel("Alpha values")
    plt.ylabel("Density")
    plt.legend(title="Alpha values")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] hop attention image saved to {save_path}")



def plot_freq_channel_attention_density(
        att_low: torch.Tensor,
        att_high: torch.Tensor,
        att_mlp: torch.Tensor,
        save_path: str = "plots/freq_att_density.png",
        figsize=(4, 3),
        dpi=150):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=figsize, dpi=dpi)
    palette = ["royalblue", "darkorange", "limegreen"]
    labels = [r"$\alpha_L$", r"$\alpha_H$", r"$\alpha_I$"]

    att_low = att_low.squeeze(-1).detach().cpu().numpy()
    att_high = att_high.squeeze(-1).detach().cpu().numpy()
    att_mlp = att_mlp.squeeze(-1).detach().cpu().numpy()

    for i, (att, color, label) in enumerate(zip([att_low, att_high, att_mlp], palette, labels)):
        sns.kdeplot(
            att, fill=True, linewidth=1.5,
            color=color, label=label
        )

    plt.xlabel("Alpha values")
    plt.ylabel("Density")
    plt.legend(title="Channel")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] frequency channel attention density image saved to {save_path}")

# plot_tsne_to_file(best_emb, labels, save_name=f"{args.dataset_name}_tSNE.png", perplexity=40)
# plot_tsne_to_file(features, labels, save_name=f"{args.dataset_name}_raw_tSNE.png", perplexity=40)
# plot_hop_attention(best_att, save_path=f"plots/{args.dataset_name}a_hop_att.png")
# plot_hop_attention_density(att, save_path=f"plots/{args.dataset_name}_hop_att_density.png")
# plot_freq_channel_attention_density(att[0], att[1], att[2], save_path=f"plots/{args.dataset_name}_freq_att_density.png")