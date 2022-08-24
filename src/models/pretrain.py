import torch.nn
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATConv, GraphConv, to_hetero

class CNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
        self.dimension = config['dimension']
        self.conv_out_channel = [32, 64, 96]
        self.conv_kernels = [4, 6, 8]
        self.encoding = encoding
        if self.encoding == 'drug':
            self.num_smile_char = 65
            self.embedding = nn.Embedding(self.num_smile_char, self.dimension, padding_idx=0)
        else:
            self.num_amino_char = 26
            self.embedding = nn.Embedding(self.num_amino_char, self.dimension, padding_idx=0)
        self.conv = 40
        self.cnns = nn.Sequential(
            nn.Conv1d(in_channels=self.dimension, out_channels=self.conv, kernel_size=self.conv_kernels[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.conv_kernels[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.conv_kernels[2]),
            nn.ReLU(),
        )
        self.cnn_dim = self.get_cnn_shape()
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_dim, self.dimension),
        )

    def get_cnn_shape(self):
        batch_size = 32
        if self.encoding == 'drug':
            max_char = self.num_smile_char
            max_len = 100
        else:
            max_char = self.num_amino_char
            max_len = 1000
        x = torch.randint(low=0, high=max_char, size=(batch_size, max_len))
        embed_ = self.embedding(x).permute(0, 2, 1)
        return self.cnns(embed_).data.view(batch_size, -1).size(1)


    def forward(self, input):
        x = self.embedding(input).permute(0, 2, 1)
        x = self.cnns(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GNNEncoder(torch.nn.Module):
    def __init__(self, model_name, hidden_channels, out_channels):
        super().__init__()

        if model_name == 'GraphSAGE':
            model = SAGEConv
        elif model_name == 'GAT':
            model = GATConv
        elif model_name == 'GraphConv':
            model = GraphConv
        else:
            assert False, "Invalid Model Name"
        self.conv1 = model((-1, -1), hidden_channels)
        self.conv2 = model((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, relations):
        super().__init__()

        self.relations = relations

        self.relations2idx = {rel: i for i, rel in enumerate(relations)}

        self.decoder = torch.nn.ModuleDict({
            rel[0]+'_'+rel[2]: torch.nn.Sequential(
                Linear(2 * hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, 1)
            )
        for rel in self.relations
        })



    def forward(self, z_dict, edge_label_index_dict):

        z = {}
        for key, edge_label_index  in edge_label_index_dict.items():
            decoder_rel2idx = key[0] +'_'+ key[2]
            row, col = edge_label_index
            z[key] = torch.cat([z_dict[key[0]][row], z_dict[key[2]][col]], dim=-1)
            z[key] = self.decoder[decoder_rel2idx](z[key])
        return z


class PretrainGNN(torch.nn.Module):
    def __init__(self, encoder_name, hidden_channels, relations, data):
        super().__init__()
        config = {'cnn_drug_filters': [32, 64, 96],
                  'cnn_drug_kernels': [4, 6, 8],
                  'dimension': hidden_channels,
                  'cnn_target_filters': [32, 64, 96],
                  'cnn_target_kernels': [4, 5, 6],
                  }
        if 'smiles' in data:
            self.feat_encoder = CNN(encoding='drug', **config)
        else:
            self.feat_encoder = CNN(encoding='protein', **config)

        self.encoder = GNNEncoder(encoder_name, hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels, relations)

    def forward(self, x_dict, edge_index_dict, edge_label_index, data):
        feat = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'smiles' in data.keys:
            str_seq = data['drug']['smiles']
            key = 'drug'
        else:
            str_seq = data['protein']['sequence']
            key = 'protein'
        for indx in range(0, len(str_seq), 32):
            batch = str_seq[indx:indx+32].to(device)
            feat.extend(self.feat_encoder(batch))
        feat = torch.stack(feat)
        z_dict = self.encoder(x_dict, edge_index_dict)
        z_dict[key] = z_dict[key] + feat
        return z_dict, self.decoder(z_dict, edge_label_index)