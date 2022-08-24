import argparse
import torch
from models.pretrain import PretrainGNN
from data_util_kg import get_kg_data
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import numpy as np
import pandas as pd

torch.manual_seed(0)
model_name = 'GraphSAGE'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--kg',
                    type=str,
                    default='drug',
                    choices=['drug','protein'],
                    help='Default kg is drug')
args = parser.parse_args()

dataset = get_kg_data(args.kg)
dataset = dataset.to(device)

for key in dataset.metadata()[1]:
    if 'rev' in key[1]:
        del dataset[key].edge_label

if args.kg == 'protein':
    edge_types = [key for key in dataset.metadata()[1] if 'rev' not in key[1]]
    rev_edge_types = [('disease', 'rev_prodis', 'protein'), ('protein', 'propro', 'protein')]
else:
    edge_types = [key for key in dataset.metadata()[1] if 'rev' not in key[1]]
    rev_edge_types = [key for key in dataset.metadata()[1] if key[0] == key[2] or 'rev' in key[1]]

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.2,
    num_test=0.2,
    neg_sampling_ratio=10.0,
    edge_types=edge_types,
    rev_edge_types=rev_edge_types,

)(dataset)



relations = [key for key in dataset.metadata()[1] if args.kg in key[0] and 'rev' not in key[1]]
model = PretrainGNN(encoder_name=model_name, hidden_channels=128, relations=relations, data=dataset).to(device)

with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

def compute_loss(pred, target):
    loss = 0
    for key, value in pred.items():
        y_hat = pred[key].squeeze(1)
        y = target[key]
        loss += loss_fn(y_hat, y)
    return loss


@torch.no_grad()
def compute_scores(pred, target):
    aupr, auroc = 0, 0
    for key, value in pred.items():
        y_hat = pred[key].sigmoid().squeeze(1).cpu().numpy()
        y = target[key].cpu().numpy().astype(int)
        aupr += average_precision_score(y, y_hat)
        auroc += roc_auc_score(y, y_hat)


    aupr /= len(pred)
    auroc /= len(pred)
    return aupr, auroc




def train():
    model.train()
    optimizer.zero_grad()
    edge_label_index = {rel: train_data[rel[0], rel[2]].edge_label_index for rel in relations}


    _, pred = model(train_data.x_dict, train_data.edge_index_dict,
                 edge_label_index, train_data)
    target = {rel: train_data[rel[0], rel[2]].edge_label.float() for rel in relations}
    loss = compute_loss(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    edge_label_index = {rel: data[rel[0], rel[2]].edge_label_index for rel in relations}
    _, pred = model(data.x_dict, data.edge_index_dict,
                 edge_label_index, test_data)
    target = {rel: data[rel[0], rel[2]].edge_label.float() for rel in relations}
    loss = compute_loss(pred, target)
    aupr, auroc = compute_scores(pred, target)
    return float(loss), float(aupr), float(auroc)

best_aupr = 0.0
patience = 0
save_path = f'saved/pretrain/{args.kg}/{model_name}'

def save():

    if not os.path.exists(save_path): os.makedirs(save_path)

    results_path = f'{save_path}/results.txt'
    pd.DataFrame([[test_aupr, test_auroc]], columns=['AUPR', 'AUROC']).to_csv(results_path, sep='\t', index=False)

    torch.save(model, f'{save_path}/checkpoint.model')

    print(f'Model saved to {save_path}')


for epoch in range(1, 100):
    loss = train()
    train_loss, train_aupr, train_auroc = test(train_data)
    val_loss, val_aupr, val_auroc = test(val_data)
    test_loss, test_aupr, test_auroc = test(test_data)
    if best_aupr <= val_aupr:
        best_aupr = val_aupr
        save()
        patience = 0
    else:
        patience += 1
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_loss:.4f}, '
          f'Val: {val_loss:.4f}, Test: {test_loss:.4f}')
    print(f'AUPR: Train: {train_aupr:.4f}, '
          f'Val: {val_aupr:.4f}, Test: {test_aupr:.4f}')
    print(f'AUROC: Train: {train_auroc:.4f}, '
          f'Val: {val_auroc:.4f}, Test: {test_auroc:.4f}')
    print('='*20)

    if patience >= 10:
        break




x_dict = dataset.x_dict

keys = [key for key in dataset.metadata()[1] if key[0] == 'drug']

model = torch.load(f'{save_path}/checkpoint.model')

def get_edge_sample(node_idx=0):
    edge_sample_dict = {}
    for key in keys:
        edge_sample = dataset.edge_index_dict[key]
        idx_ = torch.where(edge_sample[0] == node_idx)[0]
        edge_sample = edge_sample[:,idx_]
        edge_sample_dict[key] = edge_sample

    return edge_sample_dict


embedding, _ = model(dataset.x_dict, dataset.edge_index_dict, get_edge_sample(0), dataset)
embedding = embedding[args.kg].detach().cpu().numpy()
with open(f'{save_path}/embedding.npy', 'wb') as f:
    np.save(f, embedding)

print('saved embedding!')
