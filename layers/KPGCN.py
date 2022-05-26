"""
KP-GNN GCN layer
"""
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .layer_utils import degree
from .combine import *
from torch_geometric.utils import add_self_loops
from .mol_encoder import BondEncoder


class KPGCNConv(MessagePassing):
    """
    KP-GNN with GCN kernel
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        num_hop1_edge(int): number of edge type at 1 hop, need to be equal or larger than 3. default is 3.
                            Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                            larger than 2 means edge features if have.
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
        combine(str): combination method for information in different hop. select from(geometric, attention)
    """
    def __init__(self,input_size,output_size,K,num_hop1_edge=3,num_hopk_edge=3,combine="geometric"):
        super(KPGCNConv, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        assert output_size%K==0
        self.output_dk=output_size//K

        self.hop_proj=nn.Linear(input_size,output_size)

        # edge embedding for 1-hop and k-hop
        # Notice that in hops larger than one, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features like path encoding

        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge, self.output_dk,padding_idx=0)
        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, self.output_dk,padding_idx=0)
            self.combine_proj = nn.Linear(self.output_dk,output_size)
            if combine == "attention":
                self.combine = AttentionCombine(self.output_dk, self.K)
            elif combine == "geometric":
                self.combine = GeometricCombine(self.K)
            else:
                raise ValueError("Not implemented combine function")

        else:
            self.hopk_edge_emb=None
            self.combine=torch.squeeze
            self.combine_proj=nn.Identity()
        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.hop_proj.apply(self.weights_init)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)


    def forward(self,x,edge_index,edge_attr,peripheral_attr=None):

        batch_num_node=x.size(0)

        #add self loops in the edge space and update edge attr
        edge_index,_ = add_self_loops(edge_index, num_nodes = batch_num_node)
        # add features corresponding to self-loop edges, set as 1.
        self_loop_attr = torch.ones([x.size(0),self.K], dtype=torch.long)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0) # E * K


        #inital projection for each hop
        x = self.hop_proj(x) # N * H
        x=x.view(-1,self.K,self.output_dk) # N * K * dk

        #embedding of edge
        e1_emb = self.hop1_edge_emb(edge_attr[:,:1]) # E * 1 * dk
        if self.K>1:
            ek_emb = self.hopk_edge_emb(edge_attr[:,1:]) # E * K-1 * dk
            e_emb = torch.cat([e1_emb,ek_emb],dim=-2) # E * K * dk
        else:
            e_emb=e1_emb

        row, col = edge_index
        deg = degree(col, x.size(0), edge_attr) # N * K
        deg_inv_sqrt = deg.pow(-0.5)  # N * K
        norm = deg_inv_sqrt[row,...] * deg_inv_sqrt[col,...] # E * K
        x=self.propagate(edge_index, x=x, norm=norm,edge_attr=e_emb, mask=edge_attr) # N * K * dk

        #add peripheral subgraph information
        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            se_emb.masked_fill_(peripheral_attr.unsqueeze(-1)==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum((peripheral_attr>0).int(),dim=-1).unsqueeze(-1) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x=x+se_emb

        #combine
        x=self.combine_proj(self.combine(x))

        return x


    def message(self, x_j,edge_attr,norm,mask):
        x_j=norm.unsqueeze(-1)*(x_j+edge_attr) # E * K * H
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return F.relu(aggr_out)


