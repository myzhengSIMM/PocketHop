import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=2,num_features_xd=31, num_features_xt=31,hid_dim=128,
                 n_filters=31, embed_dim=256, output_dim=256, dropout=0.1):

        super(GINConvNet, self).__init__()

        dim = 31
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        # graph input feed-forward
        data_a,data_b = data[0],data[1]
        # PDB_A
        x, edge_index, batch = data_a.x, data_a.edge_index, data_a.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # PDB_B
        x_t, edge_index_t, batch_t = data_b.x, data_b.edge_index, data_b.batch
        x_t = F.relu(self.conv1(x_t, edge_index_t))
        x_t = self.bn1(x_t)
        x_t = F.relu(self.conv2(x_t, edge_index_t))
        x_t = self.bn2(x_t)
        x_t = F.relu(self.conv3(x_t, edge_index_t))
        x_t = self.bn3(x_t)
        x_t = F.relu(self.conv4(x_t, edge_index_t))
        x_t = self.bn4(x_t)
        x_t = F.relu(self.conv5(x_t, edge_index_t))
        x_t = self.bn5(x_t)
        x_t = global_add_pool(x_t, batch_t)
        x_t = F.relu(self.fc1_xd(x_t))
        x_t = F.dropout(x_t, p=0.2, training=self.training)

        # concat
        xc = x + x_t
        # xc = torch.cat((x, x_t), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
