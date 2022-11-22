"""
Graph substructure counting dataset
"""
import numpy as np
import scipy.io as sio
import torch
# from math import comb
from scipy.special import comb
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data


class GraphCountDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphCountDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        a = sio.loadmat(self.raw_paths[0])
        self.train_idx = torch.from_numpy(a['train_idx'][0])
        self.val_idx = torch.from_numpy(a['val_idx'][0])
        self.test_idx = torch.from_numpy(a['test_idx'][0])

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        b = self.processed_paths[0]
        a = sio.loadmat(self.raw_paths[0])  # 'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A = a['A'][0]
        # list of output
        Y = a['F']

        data_list = []
        for i in range(len(A)):
            a = A[i]
            A2 = a.dot(a)
            A3 = A2.dot(a)
            tri = np.trace(A3) / 6
            tailed = ((np.diag(A3) / 2) * (a.sum(0) - 2)).sum()
            cyc4 = 1 / 8 * (np.trace(A3.dot(a)) + np.trace(A2) - 2 * A2.sum())
            cus = a.dot(np.diag(np.exp(-a.dot(a).sum(1)))).dot(a).sum()

            deg = a.sum(0)
            star = 0
            for j in range(a.shape[0]):
                star += comb(int(deg[j]), 3)

            expy = torch.tensor([[tri, tailed, star, cyc4, cus]])

            E = np.where(A[i] > 0)
            edge_index = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
            x = torch.ones(A[i].shape[0], 1).long()  # change to category
            # y=torch.tensor(Y[i:i+1,:])
            data_list.append(Data(edge_index=edge_index, x=x, y=expy))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
