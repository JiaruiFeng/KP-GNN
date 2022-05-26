"""
Framework for node classification
"""
import torch.nn as nn

class NodeClassification(nn.Module):
    def __init__(self,embedding_model,output_size):
        """framework for node classification
        Args:
            embedding_model(nn.Module):  graph neural network embedding model
            output_size(int): output size, equal to the number of class for classification
        """
        super(NodeClassification, self).__init__()
        self.embedding_model=embedding_model
        hidden_size=embedding_model.hidden_size
        self.JK=self.embedding_model.JK
        self.num_layer=self.embedding_model.num_layer

        #classifier
        if self.JK=="concat":
            self.classifier = nn.Linear(hidden_size*(self.num_layer+1), output_size)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)

        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def reset_parameters(self):
        self.embedding_model.reset_parameters()
        self.classifier.apply(self.weights_init)


    def forward(self,data):
        # node representation
        x = self.embedding_model(data)
        return self.classifier(x)
