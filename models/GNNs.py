"""
General GNN framework
"""
import torch
import torch.nn as nn
from copy import  deepcopy as c
from torch_geometric.nn import BatchNorm,LayerNorm,InstanceNorm,PairNorm,GraphSizeNorm,global_add_pool
import torch.nn.functional as F


def clones( module, N):
    """Layer clone function, used for concise code writing
    Args:
        module: the layer want to clone
        N: the time of clone
    """
    return nn.ModuleList(c(module) for _ in range(N))


class GNN(nn.Module):
    """A generalized GNN framework
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,init_emb,JK="last",norm_type="batch",virtual_node=True,residual=False,use_rd=False,drop_prob=0.1):
        super(GNN, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer.output_size
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.residual=residual
        self.virtual_node = virtual_node
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        #embedding start from 1
        self.init_proj=init_emb

        self.use_rd=use_rd
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, self.hidden_size)
        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        #gnn layer list
        self.gnns=clones(gnn_layer,num_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")


        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))


        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        if self.use_rd:
            self.rd_projection.apply(self.weights_init)
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)

        self.init_proj.reset_parameters()
        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        self.output_proj.apply(self.weights_init)
        if self.use_rd:
            self.rd_projection.apply(self.weights_init)

        for g in self.gnns:
            g.reset_parameters()


    def forward(self,*argv):
        if len(argv) == 5:
            x,edge_index,edge_attr,peripheral_attr,rd,batch= argv[0], argv[1], argv[2],argv[3],argv[4],argv[5]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index,edge_attr,batch= data.x, data.edge_index, data.edge_attr,data.batch
            if "peripheral_attr" in data:
                peripheral_attr=data.peripheral_attr
            else:
                peripheral_attr=None
            if "rd" in data:
                rd=data.rd
            else:
                rd=None
        else:
            raise ValueError("unmatched number of arguments.")
        #initial projection
        x=self.init_proj(x).squeeze()

        if self.use_rd and rd is not None:
            rd_proj=self.rd_projection(rd).squeeze()
            x=x+rd_proj

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            h=self.gnns[l](h_list[l],edge_index,edge_attr,peripheral_attr)
            h=self.norms[l](h)

            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]
            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))


        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)


class GNNOGBMol(nn.Module):
    """A generalized GNN framework for OGB molecule dataset
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): Whether to add virtual node for graph embedding
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,init_emb,JK="last",norm_type="batch",virtual_node=True,residual=False,use_rd=False,drop_prob=0.75):
        super(GNNOGBMol, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer.output_size
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.virtual_node = virtual_node
        self.residual=residual
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        self.init_proj=init_emb
        self.use_rd=use_rd
        if self.use_rd:
            self.rd_projection=nn.Linear(1,self.hidden_size)


        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        #gnn layer list
        self.gnns=clones(gnn_layer,num_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        self.output_proj.apply(self.weights_init)
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)
        if self.use_rd:
            self.rd_projection.apply(self.weights_init)

        for g in self.gnns:
            g.reset_parameters()


    def forward(self,*argv):
        if len(argv) == 7:
            x,edge_index,bond_feature,edge_attr,peripheral_attr,rd,batch= argv[0], argv[1], argv[2],argv[3],argv[4],argv[5],argv[6]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index,bond_feature,edge_attr,batch= data.x, data.edge_index,data.bond_feature,data.edge_attr,data.batch
            if "peripheral_attr" in data:
                peripheral_attr=data.peripheral_attr
            else:
                peripheral_attr=None
            if "rd" in data:
                rd=data.rd
            else:
                rd=None
        else:
            raise ValueError("unmatched number of arguments.")
        #initial projection
        x=self.init_proj(x)

        if self.use_rd and rd is not None:
            rd_proj=self.rd_projection(rd)
            x=x+rd_proj

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            h=self.gnns[l](h_list[l],edge_index,bond_feature,edge_attr,peripheral_attr)
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)


class GNNQM9(nn.Module):
    """A generalized GNN framework for QM9 molecule dataset
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        input_size(int): input size of model
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): Whether to add virtual node for graph embedding
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        use_pos(bool): whether to add additional node label
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,input_size,JK="last",norm_type="batch",virtual_node=True,residual=False,use_rd=False,use_pos=False,drop_prob=0.75):
        super(GNNQM9, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer.output_size
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.virtual_node = virtual_node
        self.residual=residual
        self.use_rd=use_rd
        self.use_pos=use_pos
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")



        self.z_embedding = nn.Embedding(1000, 8)

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)

        self.init_proj=nn.Linear(input_size,self.hidden_size)

        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        #gnn layer list
        self.gnns=clones(gnn_layer,num_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        self.reset_parameters()
    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)
        self.init_proj.apply(self.weights_init)
        nn.init.xavier_uniform_(self.z_embedding.weight.data)

        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        self.output_proj.apply(self.weights_init)

        for g in self.gnns:
            g.reset_parameters()



    def forward(self,*argv):
        if len(argv) == 9:
            x, z, edge_index, bond_feature, edge_attr, peripheral_attr, rd, pos, batch = argv[0], argv[1], argv[2], \
                                                                                            argv[3], argv[4], argv[5], \
                                                                                            argv[6], argv[7], argv[8]
        elif len(argv) == 1:
            data = argv[0]
            x, z, edge_index, bond_feature, edge_attr, batch = data.x, data.z, data.edge_index, data.bond_feature, data.edge_attr, data.batch
            if "peripheral_attr" in data:
                peripheral_attr = data.peripheral_attr
                peripheral_attr = peripheral_attr.float()
            else:
                peripheral_attr = None
            if "rd" in data:
                rd = data.rd
            else:
                rd = None

            if "pos" in data:
                pos = data.pos
            else:
                pos = None
        else:
            raise ValueError("unmatched number of arguments.")

        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and rd is not None:
            rd_proj = self.rd_projection(rd)
            z_emb += rd_proj

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x=self.init_proj(x)


        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            h=self.gnns[l](h_list[l],edge_index,bond_feature,edge_attr,peripheral_attr)
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)



class KPGINPlus(nn.Module):
    """GNN framework of KP-GIN plus layer
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): Whether to add virtual node for graph embedding
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,init_emb,JK="last",norm_type="batch",virtual_node=False,residual=False,use_rd=False,drop_prob=0.1):
        super(KPGINPlus, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer[0].output_size
        self.K=gnn_layer[-1].K
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.residual=residual
        self.virtual_node = virtual_node
        self.use_rd=use_rd
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        #embedding start from 1
        self.init_proj=init_emb


        if self.use_rd:
            self.rd_projection=nn.Linear(1,self.hidden_size)

        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))


        #gnn layer list
        self.gnns=nn.ModuleList(gnn_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))
        self.output_proj.apply(self.weights_init)
        if self.use_rd:
            self.rd_projection.apply(self.weights_init)
        self.output_proj.apply(self.weights_init)
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)

        for g in self.gnns:
            g.reset_parameters()


    def forward(self,*argv):
        if len(argv) == 6:
            x,edge_index,edge_attr,peripheral_attr,rd,batch= argv[0], argv[1], argv[2],argv[3],argv[4],argv[5]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index,edge_attr,batch= data.x, data.edge_index, data.edge_attr,data.batch
            if "peripheral_attr" in data:
                peripheral_attr=data.peripheral_attr
            else:
                peripheral_attr=None
            if "rd" in data:
                rd=data.rd
            else:
                rd=None
        else:
            raise ValueError("unmatched number of arguments.")
        #initial projection
        x=self.init_proj(x).squeeze()

        if self.use_rd and rd is not None:
            rd_proj=self.rd_projection(rd).squeeze()
            x=rd_proj+x

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )


        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]
            x_list=[]
            end=-1 if l+1<=self.K else l-self.K
            for j in range(l,end,-1):
                x_list.append(h_list[j].unsqueeze(1))
            x=torch.cat(x_list,dim=1)
            k=l+1 if l+1<=self.K else self.K
            if peripheral_attr is None:
                h=self.gnns[l](x,edge_index,edge_attr[:,:k],None)
            else:
                h=self.gnns[l](x,edge_index,edge_attr[:,:k],peripheral_attr[:,:k])
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]

            h_list.append(h)


            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))


        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)

class KPGINPlusOGBMol(nn.Module):
    """GNN framework of KP-GIN plus for OGB molecule dataset
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        init_emb(nn.Module): initial node feature encoding
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): Whether to add virtual node for graph embedding
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,init_emb,JK="last",norm_type="batch",virtual_node=True,residual=False,use_rd=False,drop_prob=0.75):
        super(KPGINPlusOGBMol, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer[0].output_size
        self.K=gnn_layer[-1].K
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.virtual_node = virtual_node
        self.residual=residual
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        self.init_proj=init_emb

        self.use_rd=use_rd
        if self.use_rd:
            self.rd_projection=nn.Linear(1,self.hidden_size)


        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        #gnn layer list
        self.gnns=nn.ModuleList(gnn_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        self.reset_parameters()
    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list.apply(self.weights_init)

        self.init_proj.reset_parameters()
        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        self.output_proj.apply(self.weights_init)

        if self.use_rd:
            self.rd_projection.apply(self.weights_init)

        for g in self.gnns:
            g.reset_parameters()


    def forward(self,*argv):
        if len(argv) == 7:
            x,edge_index,bond_feature,edge_attr,peripheral_attr,rd,batch= argv[0], argv[1], argv[2],argv[3],argv[4],argv[5],argv[6]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index,bond_feature,edge_attr,batch= data.x, data.edge_index,data.bond_feature,data.edge_attr,data.batch
            if "peripheral_attr" in data:
                peripheral_attr=data.peripheral_attr
            else:
                peripheral_attr=None
            if "rd" in data:
                rd=data.rd
            else:
                rd=None
        else:
            raise ValueError("unmatched number of arguments.")
        #initial projection
        x=self.init_proj(x)
        if self.use_rd and rd is not None:
            rd_proj=self.rd_projection(rd)
            x=x+rd_proj

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]

            x_list=[]
            end=-1 if l+1<=self.K else l-self.K
            for j in range(l,end,-1):
                x_list.append(h_list[j].unsqueeze(1))
            x=torch.cat(x_list,dim=1)
            k=l+1 if l+1<=self.K else self.K
            if peripheral_attr is None:
                h=self.gnns[l](x,edge_index,bond_feature,edge_attr[:,:k],None)
            else:
                h=self.gnns[l](x,edge_index,bond_feature,edge_attr[:,:k],peripheral_attr[:,:k])
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)





class KPGINPlusQM9(nn.Module):
    """GNN framework of KPGIN plus for QM9 dataset
    Args:
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        input_size(int): input size of model
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        virtual_node(bool): Whether to add virtual node for graph embedding
        residual(bool): whether to add residual connection
        use_rd(bool): whether to add resitance distance as additional feature
        use_pos(bool): whether to add additional node label
        drop_prob (float): dropout rate
    """
    def __init__(self,num_layer,gnn_layer,input_size,JK="last",norm_type="batch",virtual_node=True,residual=False,use_rd=True, use_pos=False,drop_prob=0.75):
        super(KPGINPlusQM9, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=gnn_layer[0].output_size
        self.K=gnn_layer[-1].K
        self.dropout=nn.Dropout(drop_prob)
        self.JK=JK
        self.virtual_node = virtual_node
        self.residual=residual
        self.use_rd=use_rd
        self.use_pos=use_pos
        if self.JK=="concat":
            self.output_proj = nn.Sequential(nn.Linear((self.num_layer+1)*self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))
        else:
            self.output_proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),nn.Dropout(drop_prob))


        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(self.hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        self.z_embedding = nn.Embedding(1000, 8)
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)

        self.init_proj=nn.Linear(input_size,self.hidden_size)

        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_size)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        #gnn layer list
        self.gnns=nn.ModuleList(gnn_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(self.hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(self.hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(self.hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                    torch.nn.BatchNorm1d(2 * self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                    torch.nn.BatchNorm1d(self.hidden_size),
                    torch.nn.ReLU()))

        self.reset_parameters()
    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        if self.virtual_node:
            self.mlp_virtualnode_list.apply(self.weights_init)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.init_proj.apply(self.weights_init)
        nn.init.xavier_uniform_(self.z_embedding.weight.data)
        if self.JK=="attention":
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        self.output_proj.apply(self.weights_init)


        for g in self.gnns:
            g.reset_parameters()


    def forward(self,*argv):
        if len(argv) == 9:
            x,z,edge_index,bond_feature,edge_attr,peripheral_attr,rd,pos,batch= argv[0], argv[1], argv[2],argv[3],argv[4],argv[5],argv[6],argv[7],argv[8]
        elif len(argv) == 1:
            data = argv[0]
            x,z,edge_index,bond_feature,edge_attr,batch= data.x, data.z,data.edge_index,data.bond_feature,data.edge_attr,data.batch
            if "peripheral_attr" in data:
                peripheral_attr=data.peripheral_attr
                peripheral_attr = peripheral_attr.float()
            else:
                peripheral_attr=None
            if "rd" in data:
                rd=data.rd
            else:
                rd=None
            if "pos" in data:
                pos=data.pos
            else:
                pos=None
        else:
            raise ValueError("unmatched number of arguments.")


        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and rd is not None:
            rd_proj = self.rd_projection(rd)
            z_emb += rd_proj

        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        x=self.init_proj(x)

        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.virtual_node:
                h_list[l] = h_list[l] + virtualnode_embedding[batch]

            x_list=[]
            end=-1 if l+1<=self.K else l-self.K
            for j in range(l,end,-1):
                x_list.append(h_list[j].unsqueeze(1))
            x=torch.cat(x_list,dim=1)
            k=l+1 if l+1<=self.K else self.K
            if peripheral_attr is None:
                h=self.gnns[l](x,edge_index,bond_feature,edge_attr[:,:k],None)
            else:
                h=self.gnns[l](x,edge_index,bond_feature,edge_attr[:,:k],peripheral_attr[:,:k])
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            if self.residual:
                h=h+h_list[l]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if l < self.num_layer - 1:
                    virtualnode_embedding_temp = global_add_pool(
                        h_list[l], batch
                    ) + virtualnode_embedding
                    # transform virtual nodes using MLP

                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))
                    else:
                        virtualnode_embedding = self.dropout(self.mlp_virtualnode_list[l](virtualnode_embedding_temp))

        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2*num_layer
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)

        return self.output_proj(node_representation)
