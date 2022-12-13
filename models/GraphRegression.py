"""
Framework for graph regression
"""

import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, AttentionalAggregation


class GraphRegression(nn.Module):
    def __init__(self, embedding_model, pooling_method):
        """framework for graph regression
        Args:
            embedding_model (nn.Module):  graph neural network embedding model
            pooling_method (str): graph pooling method
        """
        super(GraphRegression, self).__init__()
        self.embedding_model = embedding_model
        hidden_size = embedding_model.hidden_size
        self.JK = self.embedding_model.JK
        self.num_layer = self.embedding_model.num_layer
        self.pooling_method = pooling_method

        # Different kind of graph pooling
        if pooling_method == "sum":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        elif pooling_method == "attention":
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_size, 1))
        else:
            raise ValueError("The pooling method not implemented")

        # regressor
        self.regressor = nn.Linear(hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.regressor.reset_parameters()
        if self.pooling_method == "attention":
            self.pool.reset_parameters()

    def forward(self, data):
        batch = data.batch
        # node representation
        x = self.embedding_model(data)
        pool_x = self.pool(x, batch)
        return self.regressor(pool_x).squeeze()
