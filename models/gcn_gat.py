import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GCN_GAT(torch.nn.Module):
    def __init__(self, n_output=2, num_features_xd=32, num_features_xt=32,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GCN_GAT, self).__init__()

        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.conv2 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 2 for classification task

    def forward(self, data):
        
        data_a,data_b = data[0],data[1]
        
        x, edge_index, batch = data_a.x, data_a.edge_index, data_a.batch
        
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        
        x_t, edge_index_t, batch_t = data_b.x, data_b.edge_index, data_b.batch
        
        # print('x shape = ', x.shape)
        x_t = self.conv1(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = self.conv2(x_t, edge_index_t)
        x_t = self.relu(x_t)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x_t = torch.cat([gmp(x_t, batch_t), gap(x_t, batch_t)], dim=1)
        x_t = self.relu(self.fc_g1(x_t))
        x_t = self.dropout(x_t)
        x_t = self.fc_g2(x_t)

        # concat
        xc = torch.cat((x, x_t), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
