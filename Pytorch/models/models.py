import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GraphConvolution, MLP
from arg_parser import arg_parser
args = arg_parser()


device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
large_datasets = [
    "deezer-europe",
    "yelp-chi",
    "Penn94",
    "arxiv-year",
    "pokec",
    "snap-patents",
    "genius",
    "twitch-gamer",
    "wiki",
]


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        nnodes,
        dropout,
        model_type,
        structure_info,
        query,
        hops,
        variant=False,
        normalization = 'local_w',
        resnet = True,
        init_layers_X=1,
    ):
        super(GCN, self).__init__()
        if model_type == "acmgcnpp":
            self.mlpX = MLP(nfeat, nhid, nhid, num_layers=init_layers_X, dropout=0)
        
        if model_type == "hcs":
            self.mlp_out = MLP(nhid, nhid, nclass, num_layers=1, dropout=dropout)
            self.mlpX = MLP(nfeat, nhid, nhid, num_layers=init_layers_X, dropout=dropout)
            self.resnet = resnet

        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.structure_info, self.nlayers, self.nnodes, self.query = (
            model_type,
            structure_info,
            nlayers,
            nnodes,
            query
        )

        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
        ):
            self.gcns.append(
                GraphConvolution(
                    nfeat,
                    nhid,
                    nnodes,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.gcns.append(
                GraphConvolution(
                    1 * nhid,
                    nclass,
                    nnodes,
                    model_type=model_type,
                    output_layer=1,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )

        if (
            self.model_type.startswith("icsgnn")
        ):
            self.gcns.append(
                GraphConvolution(
                    nfeat,
                    1 * nhid,
                    nnodes,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.gcns.append(
                GraphConvolution(
                    1 * nhid,
                    2 * nhid,
                    nnodes,
                    model_type=model_type,
                    # output_layer=1,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.gcns.append(
                GraphConvolution(
                    2 * nhid,
                    nclass,
                    nnodes,
                    model_type=model_type,
                    output_layer=1,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )

        if (self.model_type.startswith("qdgnn")
            or self.model_type.startswith("coclep")
        ):
            self.gcns.append(
                GraphConvolution(
                    nfeat,
                    nfeat,
                    nnodes,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.gcns.append(
                GraphConvolution(
                    2 * nfeat,
                    nhid,
                    nnodes,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.gcns.append(
                GraphConvolution(
                    2 * nhid,
                    nhid,
                    nnodes,
                    model_type=model_type,
                    output_layer=3,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            self.mlp_out = MLP(2 * nhid, nhid, nclass, num_layers=4, dropout=0.5)
        elif self.model_type == "hcs":
            self.gcns.append(
                GraphConvolution(
                    nfeat,
                    nhid,
                    nnodes,
                    output_layer = nclass,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                    structure_info=structure_info,
                    query = query
                )
            )
            

        elif self.model_type == "acmsgc":
            self.gcns.append(GraphConvolution(nfeat, nclass, nnodes, model_type=model_type, hops = hops))

        elif self.model_type == "acmsnowball":
            for k in range(nlayers):
                self.gcns.append(
                    GraphConvolution(
                        k * nhid + nfeat, nhid, nnodes, model_type=model_type, 
                        variant=variant, normalization=normalization, resnet = resnet, hops = hops,
                    )
                )
            self.gcns.append(
                GraphConvolution(
                    nlayers * nhid + nfeat,
                    nclass,
                    nnodes,
                    model_type=model_type,
                    variant=variant,
                    normalization=normalization,
                    resnet = resnet,
                    hops = hops,
                )
            )
        self.dropout = dropout
        self.fea_param, self.xX_param = Parameter(
            torch.FloatTensor(1, 1).to(device)
        ), Parameter(torch.FloatTensor(1, 1).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        if self.model_type == "hcs":
            self.mlpX.reset_parameters()
            self.mlp_out.reset_parameters()
        elif self.model_type == "acmgcnpp":
            self.mlpX.reset_parameters()
        else:
            pass
        # pass

    def forward(self, x, adj_low, adj_high, adj_low_unnormalized):

        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmsgc"
            or self.model_type == "hcs"
            or self.model_type == "acmsnowball"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
        ):

            x = F.dropout(x, self.dropout, training=self.training)
            if self.model_type == "acmgcnpp" or self.model_type == "hcs":
                xX = F.dropout(
                    F.relu(self.mlpX(x, input_tensor=True)),
                    self.dropout,
                    training=self.training,
                )
        if self.model_type == "acmsnowball":
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(layer(x, adj_low, adj_high, adj_low_unnormalized)),
                            self.dropout,
                            training=self.training,
                        )
                    )
                else:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(
                                layer(
                                    torch.cat([x] + list_output_blocks[0:layer_num], 1),
                                    adj_low,
                                    adj_high,
                                )
                            ),
                            self.dropout,
                            training=self.training,
                        )
                    )
            return self.gcns[-1](
                torch.cat([x] + list_output_blocks, 1), adj_low, adj_high
            )


        if self.model_type == "acmsgc":
            fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized)
            return fea1

        if self.model_type == "hcs":
            if self.resnet:
                emb, att = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized) + xX
            else:
                emb, att = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized)
            emb = F.dropout((emb), self.dropout, training=self.training) 
            fea1 = self.mlp_out(emb, input_tensor=True)
            return fea1, emb, att


        fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized)  
        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
        ):
            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)

            if self.model_type == "acmgcnpp":
                fea1 = self.gcns[1](fea1 + xX, adj_low, adj_high, adj_low_unnormalized)
            else:
                fea1 = self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)
        return fea1
