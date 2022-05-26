"""
KP-GIN plus layer
"""
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .combine import *
from .mol_encoder import BondEncoder


class KPGINPlusConv(MessagePassing):
    """KP-GNN with GIN plus convolution kernel
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hop1_edge=3,num_hopk_edge=3):
        super(KPGINPlusConv, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        self.mlp=nn.Sequential(nn.Linear(input_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU(),
                                      nn.Linear(output_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU())
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([[eps for _ in range(self.K)]]))
        else:
            self.register_buffer('eps', torch.Tensor([[eps for _ in range(self.K)]]))


        # edge embedding for 1-hop and k-hop
        # Notice that in hop, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features like path encoding
        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge, input_size,padding_idx=0)


        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, input_size,padding_idx=0)
            #self.combine=GeometricCombine(self.K)
        else:
            self.hopk_edge_emb=None
            #self.combine=nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.mlp.apply(self.weights_init)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        nn.init.zeros_(self.eps)

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)


    def forward(self,x,edge_index,edge_attr,peripheral_attr=None):
        # N * K * H
        #h=x[:,0]
        e1_emb = self.hop1_edge_emb(edge_attr[:,:1]) # E * 1 * H
        if self.K > 1:
            ek_emb = self.hopk_edge_emb(edge_attr[:, 1:])  # E * K-1 * H
            e_emb = torch.cat([e1_emb, ek_emb], dim=-2)  # E * K * H
        else:
            e_emb = e1_emb

        x_n = self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr)  # N * K * H
        # add surrounding edge information

        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            se_emb.masked_fill_(peripheral_attr.unsqueeze(-1)==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum((peripheral_attr>0).int(),dim=-1).unsqueeze(-1) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x_n=x_n+se_emb

        #x_n=self.combine(x_n).squeeze() # N * H
        h=self.mlp(torch.sum((1+self.eps.unsqueeze(-1))*x_n,dim=1))
        return h

    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return F.relu(aggr_out)


class KPGINPlusConvOGBMol(MessagePassing):
    """KP-GNN with GIN plus convolution kernel for OGB molecule dataset
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hopk_edge=3):
        super(KPGINPlusConvOGBMol, self).__init__(node_dim=0)

        self.aggr="add"
        self.K=K
        self.output_size=output_size
        self.mlp=nn.Sequential(nn.Linear(input_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU(),
                                      nn.Linear(output_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU())
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([[eps for _ in range(self.K)]]))
        else:
            self.register_buffer('eps', torch.Tensor([[eps for _ in range(self.K)]]))

        # edge embedding for 1-hop and k-hop
        # Notice that in hop, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features
        # edge embedding of 1-hop for molecule
        self.hop1_edge_emb = BondEncoder(hidden_size = input_size)


        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, input_size,padding_idx=0)
            #self.combine=GeometricCombine(self.K)
        else:
            self.hopk_edge_emb=None
            #self.combine=nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.mlp.apply(self.weights_init)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        nn.init.zeros_(self.eps)


    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)


    def forward(self,x,edge_index,bond_feature,edge_attr,peripheral_attr=None):
        # N * K * H
        e1_emb = self.hop1_edge_emb(bond_feature).unsqueeze(-2) # E * 1 * H
        if self.K > 1:
            ek_emb = self.hopk_edge_emb(edge_attr[:, 1:])  # E * K-1 * H
            e_emb = torch.cat([e1_emb, ek_emb], dim=-2)  # E * K * H
        else:
            e_emb = e1_emb

        x_n = self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr)  # N * K * H
        # add surrounding edge information

        # add surrounding edge information
        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            mask=(torch.sum(peripheral_attr,dim=-1)>0).int().unsqueeze(-1) # N * K * c * E' * 1
            se_emb.masked_fill_(mask==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum(mask,dim=-2) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total# N * K * c * dk
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x_n=x_n+se_emb

        #x_n=self.combine(x_n).squeeze() # N * H
        h=self.mlp(torch.sum((1+self.eps.unsqueeze(-1))*x_n,dim=1))
        return h

    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return F.relu(aggr_out)



class KPGINPlusConvQM9(MessagePassing):
    """KP-GNN with GIN plus convolution kernel for QM9 dataset
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable

        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hopk_edge=3):
        super(KPGINPlusConvQM9, self).__init__(node_dim=0)

        self.aggr="add"
        self.K=K
        self.output_size=output_size
        self.mlp=nn.Sequential(nn.Linear(input_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU(),
                                      nn.Linear(output_size,output_size),
                                      nn.BatchNorm1d(output_size),
                                      nn.ReLU())
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([[eps for _ in range(self.K)]]))
        else:
            self.register_buffer('eps', torch.Tensor([[eps for _ in range(self.K)]]))


        # edge embedding for 1-hop and k-hop
        # Notice that in hop, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features
        # edge embedding of 1-hop for molecule
        self.hop1_edge_emb = nn.Linear(4,input_size)

        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, input_size,padding_idx=0)
            #self.combine=GeometricCombine(self.K)
        else:
            self.hopk_edge_emb=None
            #self.combine=nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        self.hop1_edge_emb.apply(self.weights_init)
        self.mlp.apply(self.weights_init)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        nn.init.zeros_(self.eps)

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)


    def forward(self,x,edge_index,bond_feature,edge_attr,peripheral_attr=None):
        # N * K * H

        e1_emb = self.hop1_edge_emb(bond_feature).unsqueeze(-2) # E * 1 * H
        if self.K > 1:
            ek_emb = self.hopk_edge_emb(edge_attr[:, 1:])  # E * K-1 * H
            e_emb = torch.cat([e1_emb, ek_emb], dim=-2)  # E * K * H
        else:
            e_emb = e1_emb

        x_n = self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr)  # N * K * H
        # add surrounding edge information

        # add surrounding edge information
        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            mask=(torch.sum(peripheral_attr,dim=-1)>0).int().unsqueeze(-1) # N * K * c * E' * 1
            se_emb.masked_fill_(mask==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum(mask,dim=-2) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total# N * K * c * dk
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x_n=x_n+se_emb

        #x_n=self.combine(x_n).squeeze() # N * H
        h=self.mlp(torch.sum((1+self.eps.unsqueeze(-1))*x_n,dim=1))
        return h

    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return F.relu(aggr_out)


