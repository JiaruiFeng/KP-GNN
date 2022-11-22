"""
attention combine and geometric combine
"""
import torch
import torch.nn as nn


class AttentionCombine(nn.Module):
    """Attention combination for K-hop message passing GNNs
    Args:
        hidden_size(int): size of hidden representation for each hop
        K(int): number of hop in model
    """

    def __init__(self, hidden_size, K):
        super(AttentionCombine, self).__init__()

        self.attention_lstm = nn.LSTM(hidden_size, K, 1, batch_first=True, bidirectional=True, dropout=0.)

        if K < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

    def reset_parameters(self):
        self.attention_lstm.reset_parameters()

    def forward(self, x):
        self.attention_lstm.flatten_parameters()
        attention_score, _ = self.attention_lstm(x)  # N * K * 2*K
        attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(-1)  # N * K  * 1
        x = torch.sum(x * attention_score, dim=1)  # N * dk
        return x


#
# class IndependentCombine(nn.Module):
#     """Learnable weight for each hop and each channel.
#     Args:
#         hidden_size(int): size of hidden representation for each hop
#         K(int): number of hop in model
#     """
#     def __init__(self,hidden_size,K):
#         super(IndependentCombine, self).__init__()
#         self.weight=nn.Parameter(torch.rand([1,K,hidden_size]))
#
#     def reset_parameters(self):
#         nn.init.xavier_normal_(self.weight)
#
#     def forward(self,x):
#         w=torch.softmax(self.weight,dim=-2)
#         x=torch.sum(x*w,dim=-2) # N * dk
#         return x
class GeometricCombine(nn.Module):
    """Geometric combination for K-hop message passing GNNs
    Args:
        hidden_size(int): size of hidden representation for each hop
        K(int): number of hop in model
    """

    def __init__(self, K, hidden_size):
        super(GeometricCombine, self).__init__()
        self.alphas = nn.Parameter(torch.Tensor([0. for _ in range(hidden_size)]))
        self.K = K
        self.hidden_size = hidden_size

    def forward(self, x):
        thetas = self.geometric_distribution()
        x = torch.sum(x * thetas, dim=-2)  # N * dk
        return x

    def reset_parameters(self):
        nn.init.zeros_(self.alphas)

    def geometric_distribution(self):
        alphas = torch.sigmoid(self.alphas)
        thetas = torch.zeros([1, self.K, self.hidden_size], device=alphas.device)
        for i in range(self.K):
            theta = alphas * (1 - alphas) ** i
            thetas[:, i, :] = theta

        thetas = torch.softmax(thetas, dim=-2)
        return thetas


class GINEPlusCombine(nn.Module):
    """GINE+ combination for K-hop message passing GNNs
    Args:
        K(int): number of hop in model
    """

    def __init__(self, K):
        self.K = K
        self.eps = torch.nn.Parameter(torch.Tensor([[0. for _ in range(self.K)]]))

    def reset_parameters(self):
        nn.init.zeros_(self.eps)

    def forward(self, x):
        x = torch.sum((1 + self.eps.unsqueeze(-1)) * x, dim=1)
        return x
