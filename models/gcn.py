import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=32, num_features_xt=32, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        
        data_a,data_b = data[0],data[1]
        
        x, edge_index, batch = data_a.x, data_a.edge_index, data_a.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling
        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        
        # get protein input
        x_t, edge_index_t, batch_t = data_b.x, data_b.edge_index, data_b.batch
        x_t = self.conv1(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = self.conv2(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = self.conv3(x_t, edge_index_t)
        x_t = self.relu(x_t)
        x_t = gmp(x_t, batch_t)       # global max pooling
        # flatten
        x_t = self.relu(self.fc_g1(x_t))
        x_t = self.dropout(x_t)
        x_t = self.fc_g2(x_t)
        x_t = self.dropout(x_t)

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

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.out(x))
        return F.log_softmax(x, dim=-1)
    
        x_t = self.relu(self.fc_g1(x_t))
        x_t = self.dropout(x_t)
        x_t = self.fc_g2(x_t)
        x_t = self.fc1(x_t)
        x_t = self.dropout(x_t)
        x_t = self.fc2(x_t)
        x_t = self.relu(x_t)
        x_t = self.dropout(x_t)
        x_t = self.relu(self.out(x_t))
        return F.log_softmax(x_t, dim=-1)

