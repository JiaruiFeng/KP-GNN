"""
Framework for node regression
"""
import torch.nn as nn


class NodeRegression(nn.Module):
    def __init__(self, embedding_model):
        """framework for node regression
        Args:
            embedding_model (nn.Module): graph neural network embedding model
        """
        super(NodeRegression, self).__init__()
        self.embedding_model = embedding_model
        hidden_size = embedding_model.hidden_size
        self.JK = self.embedding_model.JK
        self.num_layer = self.embedding_model.num_layer
        self.regressor = nn.Linear(hidden_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.regressor.reset_parameters()

    def forward(self, data):
        # node representation
        x = self.embedding_model(data)
        return self.regressor(x).squeeze()
