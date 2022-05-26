"""
TU dataset. Adapted from NestedGNN: https://github.com/muhanzhang/NestedGNN
"""

import os
import os.path as osp
import shutil
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip,Data
from torch_geometric.io import read_tu_data
import numpy as np
import networkx as nx

class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        #we transform the edge and node feature to index and not consider the attributes
        self.data, self.slices = torch.load(self.processed_paths[0])



    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        return torch.max(self.data.x).int().item()+1

    @property
    def num_node_attributes(self):
        return 0

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        return torch.max(self.data.edge_attr).int().item()

    @property
    def num_edge_attributes(self):
        return 0

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            #data_list = [self.pre_transform(data) for data in data_list]
            new_data_list = []
            for data in tqdm(data_list):
                new_data_list.append(self.pre_transform(data))
            data_list = new_data_list
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class TUDatasetGINSplit(InMemoryDataset):
    url = 'https://github.com/weihua916/powerful-gnns/raw/master/dataset.zip'

    def __init__(self, name, root='tudatasets_gin', transform=None, pre_transform=None):
        self.name = name
        self.deg = True if name in ['IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'] else False
        super(TUDatasetGINSplit, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        train_indices, test_indices = [], []
        for i in range(10):
            train_filename = os.path.join(self.raw_dir, '10fold_idx', 'train_idx-{}.txt'.format(i + 1))
            test_filename = os.path.join(self.raw_dir, '10fold_idx', 'test_idx-{}.txt'.format(i + 1))
            train_indices.append(torch.from_numpy(np.loadtxt(train_filename, dtype=int)).to(torch.long))
            test_indices.append(torch.from_numpy(np.loadtxt(test_filename, dtype=int)).to(torch.long))

        self.train_indices = train_indices
        self.test_indices = test_indices

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return [f'{self.name}.mat', f'{self.name}.txt'] + [f'10fold_idx/test_idx-{i + 1}.txt' for i in range(10)] + [
            f'10fold_idx/train_idx-{i + 1}.txt' for i in range(10)]

    def download(self):
        folder = osp.join(self.root)
        path = download_url(self.url, folder)
        shutil.rmtree(self.raw_dir)
        extract_zip(path, folder)
        os.unlink(path)

        src = f'{self.root}/dataset'
        dest = f'{self.root}'
        print(os.listdir(self.root))
        for f in os.listdir(src):
            shutil.move(osp.join(src, f), osp.join(dest, f))
        shutil.rmtree(src)

    def process(self):
        data, num_classes = read_gin_tudataset(self.raw_dir, self.name, self.deg)
        # self.num_classes = num_classes
        data_list = [
            Data(x=datum.node_features, edge_index=datum.edge_mat, y=torch.tensor(datum.label).unsqueeze(0).long()) for
            datum in data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        """
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def read_gin_tudataset(root, dataset, degree_as_tag=False):
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}


    with open(f'{root}/{dataset}.txt' , 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)
