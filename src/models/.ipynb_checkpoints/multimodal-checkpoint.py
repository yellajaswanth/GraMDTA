import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool

from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.nn import HGTConv, Linear, HeteroConv, SAGEConv, SGConv
from torch_geometric.nn.models import AttentiveFP
import pdb

torch.manual_seed(0)


class MoleculeEncoder(torch.nn.Module):
    def __init__(self, num_features, output_dim, dropout):
        super(MoleculeEncoder, self).__init__()
        self.gcn1 = GATConv(num_features, num_features, heads=10, dropout=dropout, edge_dim=10)
        self.gcn2 = GATConv(num_features * 10, output_dim, dropout=dropout, edge_dim=10)
        self.linear = nn.Linear(output_dim, output_dim)
        self.layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.dropout(x)
        x = F.elu(self.gcn1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.elu(self.gcn2(x, edge_index, edge_attr))
        x = gmp(x, batch)
        x = self.dropout(self.linear(x))
        x = self.layer_activation(x)
        return x

class HeteroGraphEncoder(torch.nn.Module):
    def __init__(self, embedding, input_dim, output_dim, dropout):
        super(HeteroGraphEncoder, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding)
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, kg):
        x = self.embedding(kg)
        x = self.dropout(self.linear(x))
        x = self.layer_activation(x)
        return x

class ProteinSeqEncoder(torch.nn.Module):
    def __init__(self, num_features_xt, n_filters, output_dim, dropout):
        super(ProteinSeqEncoder, self).__init__()
        self.embedding_xt = nn.Embedding(num_features_xt + 1, output_dim)
        self.conv_xt = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters * 4, out_channels=n_filters * 4, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters * 4, out_channels=n_filters * 2, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters, kernel_size=7),
            # nn.Dropout(dropout)
        )
        self.cnn_dim = self.get_cnn_shape()
        self.linear = nn.Sequential(nn.Linear(self.cnn_dim, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.layer_activation = nn.ReLU()

    def get_cnn_shape(self):
        batch_size = 512
        max_char = 26
        max_len = 1000
        x = torch.randint(low=0, high=max_char, size=(batch_size, max_len))
        embed_ = self.embedding_xt(x)
        return self.conv_xt(embed_).data.view(batch_size, -1).size(1)

    def forward(self, sequence):
        x = self.embedding_xt(sequence)
        x = self.conv_xt(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.linear(x))
        x = self.layer_activation(x)
        return x

class CrossModalAttention(torch.nn.Module):
    def __init__(self, output_dim, num_heads, dropout):
        super(CrossModalAttention, self).__init__()
        self.seq2graph = nn.MultiheadAttention(output_dim, num_heads, dropout, batch_first=True)
        self.graph2seq = nn.MultiheadAttention(output_dim, num_heads, dropout, batch_first=True)

    def forward(self, modal1, modal2):
        modal1_weights, _ = self.seq2graph(modal2.unsqueeze(1), modal1.unsqueeze(1), modal1.unsqueeze(1))
        modal1_weights = modal1_weights.squeeze(1)

        modal2_weights, _ = self.graph2seq(modal1.unsqueeze(1), modal2.unsqueeze(1), modal2.unsqueeze(1))
        modal2_weights = modal2_weights.squeeze(1)

        return modal1_weights, modal2_weights


class GraMDTA(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32,  output_dim=256, dropout=0.1, drug_hetero_embed=None, protein_hetero_embed=None):
        super(KGAttentive, self).__init__()

        self.molecule_encoder = MoleculeEncoder(num_features_xd, output_dim, dropout)
        self.protein_seq_encoder = ProteinSeqEncoder(num_features_xt,n_filters,output_dim, dropout)

        self.drug_hetero_enc = HeteroGraphEncoder(embedding=drug_hetero_embed, input_dim=128, output_dim=output_dim, dropout=dropout)
        self.protein_hetero_enc = HeteroGraphEncoder(embedding=protein_hetero_embed, input_dim=128, output_dim=output_dim, dropout=dropout)

        self.cross_attention = nn.MultiheadAttention(output_dim, 4, dropout, batch_first=True)
        # combined layers
        self.fc1 = nn.Linear(output_dim*4, output_dim*4)
        self.fc2 = nn.Linear(output_dim*4, output_dim*2)
        self.fc3 = nn.Linear(output_dim*2, output_dim)
        self.out = nn.Linear(output_dim, 1)

        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.norm3 = nn.LayerNorm(output_dim)

        # activation and regularization
        self.layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, attention_map=False):
        x = self.molecule_encoder(data)
        x_kg= self.drug_hetero_enc(data.d_kg)

        xt = self.protein_seq_encoder(data.sequence)
        xt_kg = self.protein_hetero_enc(data.p_kg)

        xc = torch.cat((x.unsqueeze(1), x_kg.unsqueeze(1), xt.unsqueeze(1), xt_kg.unsqueeze(1)), 1)
        attn_wt, attention = self.cross_attention(xc, xc, xc)
        xc = xc + attn_wt
        xc = self.norm1(xc)

        xc = xc.view(xc.size(0), -1)

        xc = self.fc1(xc)
        xc = self.layer_activation(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.layer_activation(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.layer_activation(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        if attention_map:
            return out, attention
        return out
