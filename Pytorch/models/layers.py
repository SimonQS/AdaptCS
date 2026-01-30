import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from arg_parser import arg_parser
from torch_sparse import spspmm
from torch_geometric.nn import HypergraphConv
import torch.nn as nn

args = arg_parser()
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def elementwise_sparse_mul(sp_a, sp_b):
    if isinstance(sp_b, (int, float)):
        if sp_a._nnz() == 0:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0, dtype=sp_a.dtype),
                size=sp_a.size(),
                device=sp_a.device
            )
        sp_b = torch.sparse_coo_tensor(
            indices=sp_a.indices(),
            values=sp_a.values() * sp_b,  
            size=sp_a.size(), 
            device=sp_a.device
        )
    elif isinstance(sp_b, torch.Tensor) and sp_b.layout == torch.strided:
        sp_b = sp_b.to_sparse()
    
    if sp_a.device != sp_b.device:
        sp_b = sp_b.to(sp_a.device)
    
    if sp_a.size() != sp_b.size():
        raise ValueError(f"形状不匹配: {sp_a.size()} vs {sp_b.size()}")
    
    sp_a = sp_a.coalesce()
    sp_b = sp_b.coalesce()
    
    a_indices = sp_a.indices()
    a_values = sp_a.values()
    b_indices = sp_b.indices()
    b_values = sp_b.values()
    
    a_coord_dict = {(a_indices[0,i].item(), a_indices[1,i].item()): a_values[i]
                    for i in range(a_indices.size(1))}
    b_coord_dict = {(b_indices[0,i].item(), b_indices[1,i].item()): b_values[i]
                    for i in range(b_indices.size(1))}
    
    common_coords = a_coord_dict.keys() & b_coord_dict.keys()
    
    if not common_coords:
        return torch.sparse_coo_tensor(
            indices=torch.empty((2, 0), dtype=torch.long),
            values=torch.empty(0, dtype=sp_a.dtype),
            size=sp_a.size(),
            device=sp_a.device
        )
    
    rows, cols = zip(*common_coords)
    result_indices = torch.stack([
        torch.tensor(rows, device=sp_a.device),
        torch.tensor(cols, device=sp_a.device)
    ])
    
    result_values = torch.tensor(
        (a_coord_dict[(r,c)] * b_coord_dict[(r,c)] 
         for r,c in common_coords),
        dtype=sp_a.dtype,
        device=sp_a.device
    )
    
    return torch.sparse_coo_tensor(
        indices=result_indices,
        values=result_values,
        size=sp_a.size(),
        device=sp_a.device
    ).coalesce()


class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        nnodes,
        model_type,
        hops,
        query,
        output_layer=0,
        variant=False,
        normalization = 'local_w',
        resnet = True,
        structure_info=0,
    ):
        super(GraphConvolution, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.model_type,
            self.structure_info,
            self.variant,
            self.normalization,
            self.resnet,
            self.hops,
            self.query
        ) = (
            in_features,
            out_features,
            output_layer,
            model_type,
            structure_info,
            variant,
            normalization,
            resnet,
            hops,
            query,
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
        )
        self.weight_query = Parameter(torch.FloatTensor(nnodes, out_features).to(device))
        self.graph_hgc = HypergraphConv(in_features, out_features)
        self.query_hgc = HypergraphConv(nnodes, out_features)
        self.coclep_mlp = Parameter(torch.FloatTensor(out_features*4, out_features*2).to(device))
    
        self.weight_low_channel, self.weight_high_channel= (
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
        )
        self.weight_low_channel_2, self.weight_high_channel_2= (
            Parameter(torch.FloatTensor(in_features * 2, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features * 2, out_features).to(device)),
        )

        self.weight_low_mix, self.weight_high_mix= (
            Parameter(torch.FloatTensor(out_features*2, out_features).to(device)),
            Parameter(torch.FloatTensor(out_features*2, out_features).to(device)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
        )

        self.att_vec_low_2, self.att_vec_high_2, self.att_vec_mlp_2 = (
            Parameter(torch.FloatTensor(2 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(2 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(2 * out_features, 1).to(device)),
        )


        self.att_node_wise_low, self.att_node_wise_high = (
            Parameter(torch.FloatTensor(in_features).to(device)),
            Parameter(torch.FloatTensor(in_features).to(device))
        )

        self.att_hopwise_shared= (
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device))
        )
        self.att_hopwise_distinct= (
            Parameter(torch.FloatTensor(1 * out_features, hops).to(device))
        )        
        self.hopwise_fuse_mlp= (
            Parameter(torch.FloatTensor(hops * out_features, out_features).to(device))
        )
        self.att_hopwise_bank= (
            Parameter(torch.FloatTensor(1 * out_features, output_layer).to(device))
        )

        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_inter = nn.LayerNorm(out_features)

        
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1).to(device)
        )
        self.struc_low = Parameter(torch.FloatTensor(nnodes, out_features).to(device))
        self.att_vec_2 = Parameter(torch.FloatTensor(2, 2).to(device))
        if self.structure_info == 0:
            self.att_vec = Parameter(torch.FloatTensor(3, 3).to(device))
        else:
            self.att_vec = Parameter(torch.FloatTensor(4, 4).to(device))

        self.att_vec_hop = Parameter(torch.FloatTensor(hops, hops).to(device))

        self.q_linear = torch.nn.Linear(out_features, out_features).to(device)  # Query linear transformation
        self.k_linear = torch.nn.Linear(out_features, out_features).to(device)  # Key linear transformation
        self.v_linear = torch.nn.Linear(out_features, out_features).to(device)  # Value linear transformation
        self.scale = torch.sqrt(torch.FloatTensor([hops])).to(device)

        self.reset_parameters()


    def reset_parameters(self):
        for param in [
            self.weight_low, self.weight_high, self.weight_mlp, self.struc_low,
            self.weight_query, self.coclep_mlp,
            self.weight_low_channel, self.weight_high_channel,
            self.weight_low_channel_2, self.weight_high_channel_2,
            self.weight_low_mix, self.weight_high_mix,
            self.hopwise_fuse_mlp
        ]:
            if param.dim() >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            else:
                nn.init.uniform_(param, -0.1, 0.1)

        for param in [
            self.att_vec_high, self.att_vec_low, self.att_vec_mlp,
            self.att_struc_low, self.att_vec_high_2, self.att_vec_low_2,
            self.att_vec_mlp_2, self.att_hopwise_shared, self.att_hopwise_distinct,
            self.att_hopwise_bank, self.att_vec_hop, self.att_vec_2, self.att_vec,
        ]:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, -0.1, 0.1)

        for param in [self.att_node_wise_low, self.att_node_wise_high]:
            nn.init.uniform_(param, -0.1, 0.1)

        self.q_linear.reset_parameters()
        self.k_linear.reset_parameters()
        self.v_linear.reset_parameters()

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_inter.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()


    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        if self.model_type == "acmgcnp" or self.model_type == "acmgcnpp" or self.model_type == "hcs" and args.layer_norm:
            output_low, output_high, output_mlp = (
                self.layer_norm_low(output_low),
                self.layer_norm_high(output_high),
                self.layer_norm_mlp(output_mlp),
            )
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,)),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention2(self, output_low, output_high):
        T = 2
        if self.model_type in ["acmgcnp", "acmgcnpp", "hcs"] and args.layer_norm:
            output_low = self.layer_norm_low(output_low)
            output_high = self.layer_norm_high(output_high)

        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm(output_low, self.att_vec_low),
                            torch.mm(output_high, self.att_vec_high),
                        ],
                        1,
                    )
                ),
                self.att_vec_2,
            )
            / T
        )

        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None]

    def attention2_2(self, output_low, output_high):
        T = 2
        if self.model_type in ["acmgcnp", "acmgcnpp", "hcs"] and args.layer_norm:
            output_low = self.layer_norm_low(output_low)
            output_high = self.layer_norm_high(output_high)

        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm(output_low, self.att_vec_low_2),
                            torch.mm(output_high, self.att_vec_high_2),
                        ],
                        1,
                    )
                ),
                self.att_vec_2,
            )
            / T
        )

        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None]


    def attention_hop(self, channels, distinct):
        if args.fuse_hop == 'self':
            T = channels.shape[0]  
            root_channel = channels[0]  
            attention_score = []  

            if distinct:
                for i in range(T):
                    hop_channel = channels[i]
                    att_hop = torch.mm(root_channel + hop_channel, self.att_hopwise_distinct[:, i].unsqueeze(-1))
                    attention_score.append(att_hop)
            else:
                for i in range(T):
                    hop_channel = channels[i]
                    att_hop = torch.mm(root_channel + hop_channel, self.att_hopwise_shared)
                    attention_score.append(att_hop)
            
            attention_score = torch.cat(attention_score, dim=1)
            logits = torch.mm(torch.sigmoid(attention_score), self.att_vec_hop) / T
            att = torch.softmax(logits, 1)
            adaptive_w = None

        elif args.fuse_hop == 'cat':
            T = channels.shape[0]

            if distinct:
                feature_concat = torch.cat([torch.mm(channels[i], 
                                            self.att_hopwise_distinct[:,i].unsqueeze(-1)) 
                                            for i in range(T)], 1,) 
            else:
                feature_concat = torch.cat([torch.mm(channels[i], 
                                            self.att_hopwise_shared) 
                                            for i in range(T)], 1,)
            
            logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec_hop) / T
            att = torch.softmax(logits, 1)
            adaptive_w = None

        elif args.fuse_hop == 'qkv':
            T = channels.shape[0]
            queries = []
            keys = []
            values = []
            for i in range(T):
                hop_channel = channels[i]
                q = self.q_linear(hop_channel)  
                k = self.k_linear(hop_channel)  
                v = self.v_linear(hop_channel)  
                queries.append(q)
                keys.append(k)
                values.append(v)

            queries = torch.stack(queries, dim=0).to(device)  
            keys = torch.stack(keys, dim=0).to(device)        
            values = torch.stack(values, dim=0).to(device)       
            attention_scores = torch.einsum('btd,btd->btd', queries, keys) / self.scale
            attention_weights = F.softmax(attention_scores, dim=0) 
            att = (attention_weights * values).sum(dim=0)  
            adaptive_w = None

        elif args.fuse_hop == "bank":
            T          = channels.size(0)                 
            n, c_dim   = channels.size(1), channels.size(2)
            num_cls    = self.att_hopwise_bank.size(1)     
            root = channels[0]                              
            class_logits   = torch.matmul(root, self.att_hopwise_bank)
            class_weights  = torch.softmax(class_logits, dim=1)     

            adaptive_w = torch.matmul(class_weights, self.att_hopwise_bank.t()) 
            hop_scores = []                                           
            for i in range(T):
                score_i = (channels[i] * adaptive_w).sum(dim=1, keepdim=True)    
                hop_scores.append(score_i)
            hop_scores = torch.cat(hop_scores, dim=1)     

            att = torch.softmax(hop_scores, dim=1)      
        return att, adaptive_w

    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        if args.layer_norm:
            feature_concat = torch.cat(
                [
                    torch.mm(self.layer_norm_low(output_low), self.att_vec_low),
                    torch.mm(self.layer_norm_high(output_high), self.att_vec_high),
                    torch.mm(self.layer_norm_mlp(output_mlp), self.att_vec_mlp),
                    torch.mm(self.layer_norm_struc_low(struc_low), self.att_struc_low),
                ],
                1,
            )
        else:
            feature_concat = torch.cat(
                [
                    torch.mm((output_low), self.att_vec_low),
                    torch.mm((output_high), self.att_vec_high),
                    torch.mm((output_mlp), self.att_vec_mlp),
                    torch.mm((struc_low), self.att_struc_low),
                ],
                1,
            )

        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec) / T

        att = torch.softmax(logits, 1)
        return (
            att[:, 0][:, None],
            att[:, 1][:, None],
            att[:, 2][:, None],
            att[:, 3][:, None],
        )


    def distinct_hop_forward(
        self,
        args,
        input_feats,  
        adj_low,      
        adj_high,     
        adj_low_unnormalized,  
        device
    ):

        channels = []
        output_low = []
        output_high = []

        output_mlp = torch.mm(input_feats, self.weight_mlp)  

        if args.normalization == 'local_w':
            input_transformed_low = torch.mul(input_feats, self.att_node_wise_low) 
            attn_scores_low = torch.matmul(input_transformed_low, input_transformed_low.T)
            attn_scores_low = torch.sigmoid(attn_scores_low).to_sparse()

            one_tensor = torch.ones(attn_scores_low.size(), device=device)
            attn_scores_high = one_tensor - attn_scores_low

            normalized_adj_low_list = []
            normalized_adj_high_list = []
            for i in range(len(adj_low)):
                temp_low  = adj_low[i]  * attn_scores_low
                temp_high = adj_high[i] * attn_scores_high
                normalized_adj_low_list.append(temp_low)
                normalized_adj_high_list.append(temp_high)
        else:
            normalized_adj_low_list  = [adj_low[i]  for i in range(len(adj_low))]
            normalized_adj_high_list = [adj_high[i] for i in range(len(adj_high))]

        for i in range(len(adj_low)):  
            
            if self.variant == 1:
                out_low_i = torch.spmm(
                    normalized_adj_low_list[i],
                    F.relu(torch.mm(input_feats, self.weight_low_channel))
                )
                out_high_i = torch.spmm(
                    normalized_adj_high_list[i],
                    F.relu(torch.mm(input_feats, self.weight_high_channel))
                )
            else:
                out_low_i = F.relu(
                    torch.mm(
                        torch.spmm(normalized_adj_low_list[i], input_feats),
                        self.weight_low_channel
                    )
                )
                out_high_i = F.relu(
                    torch.mm(
                        torch.spmm(normalized_adj_high_list[i], input_feats),
                        self.weight_high_channel
                    )
                )

            output_low.append(out_low_i)
            output_high.append(out_high_i)

            if self.structure_info == 1:
                output_struc_low = F.relu(
                    torch.mm(adj_low_unnormalized, self.struc_low)
                )
                (
                    self.att_low,
                    self.att_high,
                    self.att_mlp,
                    self.att_struc_vec_low,
                ) = self.attention4(out_low_i, out_high_i, output_mlp, output_struc_low)

                channels.append(
                    4 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                        + self.att_mlp * output_mlp
                        + self.att_struc_vec_low * output_struc_low
                    )
                )
            elif self.structure_info == 0:
                self.att_low, self.att_high, self.att_mlp = self.attention3(
                    out_low_i, out_high_i, output_mlp
                )
                channels.append(
                    3 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                        + self.att_mlp * output_mlp
                    )
                )
            elif self.structure_info == 2:
                self.att_low, self.att_high = self.attention2(
                    out_low_i, out_high_i
                )
                channels.append(
                    2 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                    )
                )

<<<<<<< HEAD
        fused_emb, att = self._fuse_hopwise(args, channels, device) 
        return fused_emb, att
=======
        fused_emb = self._fuse_hopwise(args, channels, device) 
        return fused_emb
>>>>>>> origin/main

    def cross_skip_forward(
        self,
        args,
        low_channels,    
        high_channels,   
        input_feats,     
        adj_low_unnormalized, 
        device
    ):
        channels = []
        output_low = []
        output_high = []

        output_mlp = torch.mm(input_feats, self.weight_mlp)

        H = len(low_channels) 
        for i in range(H):
            feat_low_i  = low_channels[i]
            feat_high_i = high_channels[i]

            if self.variant == 1:
                out_low_i = F.relu(torch.mm(feat_low_i, self.weight_low_channel))
                out_high_i = F.relu(torch.mm(feat_high_i, self.weight_high_channel))
            else:
                out_low_i = F.relu(torch.mm(feat_low_i, self.weight_low_channel))
                out_high_i = F.relu(torch.mm(feat_high_i, self.weight_high_channel))

            output_low.append(out_low_i)
            output_high.append(out_high_i)

            if self.structure_info == 1:
                output_struc_low = F.relu(
                    torch.mm(adj_low_unnormalized, self.struc_low)
                )
                (self.att_low, self.att_high, self.att_mlp, self.att_struc_vec_low) = self.attention4(
                    out_low_i, out_high_i, output_mlp, output_struc_low
                )
                channels.append(
                    4 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                        + self.att_mlp * output_mlp
                        + self.att_struc_vec_low * output_struc_low
                    )
                )
            elif self.structure_info == 0:
                (self.att_low, self.att_high, self.att_mlp) = self.attention3(
                    out_low_i, out_high_i, output_mlp
                )
                channels.append(
                    3 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                        + self.att_mlp * output_mlp
                    )
                )
            elif self.structure_info == 2:
                (self.att_low, self.att_high) = self.attention2(
                    out_low_i, out_high_i
                )
                channels.append(
                    2 * (
                        self.att_low * out_low_i
                        + self.att_high * out_high_i
                    )
                )

        fused_emb, att = self._fuse_hopwise(args, channels, device)
        return fused_emb, (self.att_low, self.att_high, self.att_mlp)


    def _fuse_hopwise(self, args, channels, device):

        if args.fuse_hop == 'mlp':
            cat_channels = torch.cat(channels, dim=1).to(device)  
            fused_emb = F.relu(torch.mm(cat_channels, self.hopwise_fuse_mlp))
<<<<<<< HEAD
            att = None  # MLP fusion doesn't use attention
            return fused_emb, att
=======
            return fused_emb
>>>>>>> origin/main

        elif args.fuse_hop == 'qkv':
            stack_channels = torch.stack(channels, dim=0).to(device)
            fused_emb = self.attention_hop(stack_channels, args.att_hopwise_distinct)
<<<<<<< HEAD
            att = None  # QKV fusion returns fused result directly
            return fused_emb, att 
=======
            return fused_emb 
>>>>>>> origin/main

        elif args.fuse_hop == 'bank':
            stack_channels = torch.stack(channels, dim=0).to(device) 
            att, adaptive_w = self.attention_hop(stack_channels, args.att_hopwise_distinct) 
            att_broadcast = att.t().unsqueeze(-1) 
            adaptive_w = torch.sigmoid(adaptive_w)
            inter = stack_channels * adaptive_w
            inter = torch.sigmoid(inter)
            fused_emb = (inter * att_broadcast).sum(dim=0)  
            return fused_emb, att_broadcast

        else:
            stack_channels = torch.stack(channels, dim=0).to(device)  
            att, _ = self.attention_hop(stack_channels, args.att_hopwise_distinct) 
            att_broadcast = att.t().unsqueeze(-1) 
            fused_emb = (stack_channels * att_broadcast).sum(dim=0)   
<<<<<<< HEAD
            return fused_emb, att_broadcast
=======
            return fused_emb
>>>>>>> origin/main


    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        output = 0
        if self.model_type == "mlp":
            output_mlp = torch.mm(input, self.weight_mlp)
            return output_mlp
        elif self.model_type == "sgc" or self.model_type == "gcn":
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == "acmsgc":
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(adj_high, torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            self.att_low, self.att_high, self.att_mlp = self.attention3(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )

        elif self.model_type == "hcs":
            if args.approach == 'distinct_hop':
<<<<<<< HEAD
                fused_emb, att = self.distinct_hop_forward(
=======
                fused_emb = self.distinct_hop_forward(
>>>>>>> origin/main
                    args, input, adj_low, adj_high, adj_low_unnormalized, device
                )
            elif args.approach == 'cross_skip' or args.approach == 'distinct_hop_svds_low' or args.approach == 'distinct_hop_svds_rand':
                fused_emb, att = self.cross_skip_forward(
                    args, adj_low, adj_high, input, adj_low_unnormalized, device
                )
            else:
                fused_emb = None
<<<<<<< HEAD
                att = None
=======
>>>>>>> origin/main
            return fused_emb, att

        else:
            if self.variant:

                output_low = torch.spmm(
                    adj_low, F.relu(torch.mm(input, self.weight_low))
                )

                output_high = torch.spmm(
                    adj_high, F.relu(torch.mm(input, self.weight_high))
                )
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            else:
                output_low = F.relu(
                    torch.spmm(adj_low, (torch.mm(input, self.weight_low)))
                )
                output_high = F.relu(
                    torch.spmm(adj_high, (torch.mm(input, self.weight_high)))
                )
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            if self.model_type == "acmgcn" or self.model_type == "acmsnowball":
                self.att_low, self.att_high, self.att_mlp = self.attention3(
                    (output_low), (output_high), (output_mlp)
                )
                return 3 * (
                    self.att_low * output_low
                    + self.att_high * output_high
                    + self.att_mlp * output_mlp
                )
            else:
                if self.structure_info:
                    output_struc_low = F.relu(
                        torch.mm(adj_low_unnormalized, self.struc_low)
                    )
                    (
                        self.att_low,
                        self.att_high,
                        self.att_mlp,
                        self.att_struc_vec_low,
                    ) = self.attention4(
                        (output_low), (output_high), (output_mlp), output_struc_low
                    )
                    return 1 * (
                        self.att_low * output_low
                        + self.att_high * output_high
                        + self.att_mlp * output_mlp
                        + self.att_struc_vec_low * output_struc_low
                    )
                else:
                    self.att_low, self.att_high, self.att_mlp = self.attention3(
                        (output_low), (output_high), (output_mlp)
                    )
                    return 3 * (
                        self.att_low * output_low
                        + self.att_high * output_high
                        + self.att_mlp * output_mlp
                    )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph["node_feat"]
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
