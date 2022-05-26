"""
KP-GNN GIN layer
"""
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .combine import *
from .mol_encoder import BondEncoder


class KPGINConv(MessagePassing):
    """
    KP-GNN with GIN kernel
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable
        num_hop1_edge(int): number of edge type at 1 hop, need to be equal or larger than 3. default is 3.
                            Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                            larger than 2 means edge features if have.
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
        combine(str): combination method for information in different hop. select from(geometric, attention)
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hop1_edge=3,num_hopk_edge=3,combine="geometric"):
        super(KPGINConv, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        assert input_size%K==0
        assert  output_size%K==0
        self.input_dk=input_size//K
        self.output_dk=output_size//K
        # multi-layer perceptron
        self.hop_proj1=torch.nn.Parameter(torch.Tensor(self.K,self.input_dk,self.output_dk))
        self.hop_bias1=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))
        self.hop_proj2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk,self.output_dk))
        self.hop_bias2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))


        # edge embedding for 1-hop and k-hop
        # Notice that in hops larger than one, there is no actually edge feature, therefore need addtional embedding layer to encode
        # self defined features like path encoding

        self.hop1_edge_emb = torch.nn.Embedding(num_hop1_edge, self.input_dk,padding_idx=0)

        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, self.input_dk,padding_idx=0)
            self.combine_proj=nn.Linear(self.output_dk,output_size)
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
        nn.init.xavier_uniform_(self.hop_proj1.data)
        nn.init.xavier_uniform_(self.hop_proj2.data)
        nn.init.zeros_(self.hop_bias1.data)
        nn.init.zeros_(self.hop_bias2.data)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        self.combine_proj.apply(self.weights_init)
        nn.init.zeros_(self.eps)

    def forward(self,x,edge_index,edge_attr,peripheral_attr=None):

        x=x.view(-1,self.K,self.input_dk) # N * K * dk


        #embedding of edge
        e1_emb = self.hop1_edge_emb(edge_attr[:,:1]) # E * 1 * dk
        if self.K>1:
            ek_emb = self.hopk_edge_emb(edge_attr[:,1:]) # E * K-1 * dk
            e_emb = torch.cat([e1_emb,ek_emb],dim=-2) # E * K * dk
        else:
            e_emb=e1_emb

        x_n=self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr) # N * K * dk

        #add peripheral subgraph information
        if peripheral_attr is not None:
            se_emb = self.hop1_edge_emb(peripheral_attr) # N * K * c * E' * dk
            se_emb.masked_fill_(peripheral_attr.unsqueeze(-1)==0,0.)
            se_emb=torch.sum(se_emb,dim=-2) # N * K * c * dk
            total=torch.sum((peripheral_attr>0).int(),dim=-1).unsqueeze(-1) # N * K * c * 1
            total[total==0]=1
            se_emb=se_emb/total
            se_emb=torch.sum(se_emb,dim=-2) # N * K * dk
            x_n=x_n+se_emb

        x=x_n + (1 + self.eps) * x
        x=x.permute(1,0,2)
        x=F.relu(torch.matmul(x,self.hop_proj1)+self.hop_bias1.unsqueeze(1))
        x=F.relu(torch.matmul(x,self.hop_proj2)+self.hop_bias2.unsqueeze(1))
        x=x.permute(1,0,2)
        #combine
        x=self.combine_proj(self.combine(x))
        return x


    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return aggr_out



class KPGINConvOGBMol(MessagePassing):
    """
    KP-GNN with GIN kernel for OGB molecule dataset
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable

        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
        combine(str): combination method for information in different hop. select from(geometric, attention)
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hopk_edge=3,combine="geometric"):
        super(KPGINConvOGBMol, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        assert input_size%K==0
        assert  output_size%K==0
        self.input_dk=input_size//K
        self.output_dk=output_size//K
        # multi-layer perceptron
        self.hop_proj1=torch.nn.Parameter(torch.Tensor(self.K,self.input_dk,self.output_dk))
        self.hop_bias1=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))
        self.hop_proj2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk,self.output_dk))
        self.hop_bias2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))


        # edge embedding of 1-hop for molecule
        self.hop1_edge_emb = BondEncoder(hidden_size = self.input_dk)

        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, self.input_dk,padding_idx=0)
            self.combine_proj=nn.Linear(self.output_dk,output_size)
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


    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.combine_proj.apply(self.weights_init)
        nn.init.xavier_uniform_(self.hop_proj1.data)
        nn.init.xavier_uniform_(self.hop_proj2.data)
        nn.init.zeros_(self.hop_bias1.data)
        nn.init.zeros_(self.hop_bias2.data)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        nn.init.zeros_(self.eps)


    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)


    def forward(self,x,edge_index,bond_feature,edge_attr,peripheral_attr=None):

        #inital projection for each hop
        x=x.view(-1,self.K,self.input_dk) # N * K * dk

        #embedding of edge
        #E * K * 3
        e1_emb = self.hop1_edge_emb(bond_feature).unsqueeze(-2) # E * 1 * dk
        if self.K>1:
            ek_emb = self.hopk_edge_emb(edge_attr[:,1:]) # E * K-1 * dk
            e_emb = torch.cat([e1_emb,ek_emb],dim=-2) # E * K * dk
        else:
            e_emb=e1_emb

        x_n=self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr) # N * K * dk
        #add peripheral subgraph information
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


        x=x_n + (1 + self.eps) * x
        x=x.permute(1,0,2)
        x=F.relu(torch.matmul(x,self.hop_proj1)+self.hop_bias1.unsqueeze(1))
        x=F.relu(torch.matmul(x,self.hop_proj2)+self.hop_bias2.unsqueeze(1))
        x=x.permute(1,0,2)
        #combine
        x=self.combine_proj(self.combine(x))
        return x


    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return aggr_out




class KPGINConvQM9(MessagePassing):
    """
    KP-GNN with GIN kernel for QM9 dataset
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        K(int): number of hop to consider in Convolution layer
        eps(float): initial epsilon
        train_eps(bool):whether the epsilon is trainable
        num_hopk_edge(int): number of edge type higher than 1 hop, need to be equal or larger than 3. default is 3.
                    Where index 0 represent mask(no edge), index 1 represent self-loop, index 2 represent edge.
                    larger than 2 means edge features if have.
        combine(str): combination method for information in different hop. select from(geometric, attention)
    """
    def __init__(self,input_size,output_size,K,eps=0.,train_eps=False,num_hopk_edge=3,combine="geometric"):
        super(KPGINConvQM9, self).__init__(node_dim=0)
        self.aggr="add"
        self.K=K
        self.output_size=output_size
        assert input_size%K==0
        assert  output_size%K==0
        self.input_dk=input_size//K
        self.output_dk=output_size//K
        # multi-layer perceptron
        self.hop_proj1=torch.nn.Parameter(torch.Tensor(self.K,self.input_dk,self.output_dk))
        self.hop_bias1=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))
        self.hop_proj2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk,self.output_dk))
        self.hop_bias2=torch.nn.Parameter(torch.Tensor(self.K,self.output_dk))

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        # edge embedding of 1-hop for molecule
        self.hop1_edge_emb = BondEncoder(hidden_size = self.input_dk)
        #If K larger than 1, define additional embedding and combine function
        if self.K>1:
            self.hopk_edge_emb = torch.nn.Embedding(num_hopk_edge, self.input_dk,padding_idx=0)
            self.combine_proj=nn.Linear(self.output_dk,output_size)
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


    def reset_parameters(self):
        self.hop1_edge_emb.reset_parameters()
        self.combine_proj.apply(self.weights_init)
        nn.init.xavier_uniform_(self.hop_proj1.data)
        nn.init.xavier_uniform_(self.hop_proj2.data)
        nn.init.zeros_(self.hop_bias1.data)
        nn.init.zeros_(self.hop_bias2.data)
        if self.hopk_edge_emb is not None:
            nn.init.xavier_uniform_(self.hopk_edge_emb.weight.data)
        nn.init.zeros_(self.eps)


    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)


    def forward(self,x,edge_index,bond_feature,edge_attr,peripheral_attr=None):

        #inital projection for each hop
        x=x.view(-1,self.K,self.input_dk) # N * K * dk
        #embedding of edge
        #E * K * 3
        e1_emb = self.hop1_edge_emb(bond_feature).unsqueeze(-2) # E * 1 * dk
        if self.K>1:
            ek_emb = self.hopk_edge_emb(edge_attr[:,1:]) # E * K-1 * dk
            e_emb = torch.cat([e1_emb,ek_emb],dim=-2) # E * K * dk
        else:
            e_emb=e1_emb

        x_n=self.propagate(edge_index, x=x, edge_attr=e_emb, mask=edge_attr) # N * K * dk
        #add peripheral subgraph information
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


        x=x_n + (1 + self.eps) * x
        x=x.permute(1,0,2)
        x=F.relu(torch.matmul(x,self.hop_proj1)+self.hop_bias1.unsqueeze(1))
        x=F.relu(torch.matmul(x,self.hop_proj2)+self.hop_bias2.unsqueeze(1))
        x=x.permute(1,0,2)
        #combine
        x=self.combine_proj(self.combine(x))
        return x


    def message(self, x_j,edge_attr,mask):
        x_j=x_j+edge_attr # E * K * dk
        mask=mask.unsqueeze(-1) # E * K * 1
        return x_j.masked_fill_(mask==0, 0.)


    def update(self,aggr_out):
        return aggr_out

