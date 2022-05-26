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
    def __init__(self,hidden_size,K):
        super(AttentionCombine, self).__init__()

        self.attention_lstm = nn.LSTM(hidden_size,K, 1, batch_first=True, bidirectional=True, dropout=0.)
        for layer_p in self.attention_lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        if K < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

    def forward(self,x):
        self.attention_lstm.flatten_parameters()
        attention_score, _ = self.attention_lstm(x)  # N * K * 2*K
        attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(-1)  # N * K  * 1
        x= torch.sum(x * attention_score, dim=1) # N * dk
        return x

class GeometricCombine(nn.Module):
    """Geometric combination for K-hop message passing GNNs
    Args:
        hidden_size(int): size of hidden representation for each hop
        K(int): number of hop in model
    """
    def __init__(self,K):
        super(GeometricCombine, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.]))
        self.K=K

    def forward(self,x):
        thetas=self.geometric_distribution()
        x=torch.sum(x*thetas,dim=-2) # N * dk
        return x

    def geometric_distribution(self):
        alpha=torch.sigmoid(self.alpha)
        thetas=torch.zeros([1,self.K,1],device=alpha.device)
        for i in range(self.K):
            theta=self.alpha*(1-self.alpha)**i
            thetas[:,i,:]=theta

        thetas=torch.softmax(thetas,dim=-2)
        return thetas

