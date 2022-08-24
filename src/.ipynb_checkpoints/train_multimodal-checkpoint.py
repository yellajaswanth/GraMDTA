import torch
from torch.utils.data import DataLoader

from models.multimodal import GraMDTA

import warnings
warnings.filterwarnings("ignore")
from metrics import eval_graphDTA
from argparser import parse_args
import pytorch_warmup as warmup
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_util_multimodal import *
import pickle as pkl

args = parse_args()

# Argparser
LR = args.lr
NUM_EPOCHS = args.epochs
EPOCH_PATIENCE = args.patience


def train(model_name, UPSAMPLE, EXP_TYPE):


    # Data loaders
    path = f'../data/drugbank/{UPSAMPLE}/{EXP_TYPE}/'
    train = HeteroMoleculeNet(path, 'train', pre_transform=HeteroGenFeatures()).shuffle()
    val = HeteroMoleculeNet(path, 'val', pre_transform=HeteroGenFeatures()).shuffle()
    test = HeteroMoleculeNet(path, 'test', pre_transform=HeteroGenFeatures())



    # make data PyTorch mini-batch processing ready
    train_data_loader = DataLoader(train, batch_size=512, shuffle=True)
    val_data_loader = DataLoader(val, batch_size=512, shuffle=False)
    test_data_loader = DataLoader(test, batch_size=512, shuffle=False)

    train_labels = []
    for data in train_data_loader:
        train_labels.extend([data.y.flatten()[0].item()])

    train_labels = torch.tensor(train_labels)
    num_pos_examples = torch.sum(train_labels, dtype=torch.float)
    num_neg_examples = len(train_labels) - num_pos_examples
    pos_weights = num_neg_examples/num_pos_examples


    model = load_model()
    device = torch.device('cuda')
    model = model.to(device)

    # MODEL_NAME = '{}'.format(type(model).__name__)
    MODEL_NAME = model_name
    print(MODEL_NAME)
    # print('{} million parameters'.format(round(sum(p.numel() for p in model.parameters()) * 1e-6, 3)))

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    num_steps = len(train_data_loader) * NUM_EPOCHS
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)
    warmup_scheduler = warmup.UntunedLinearWarmup(optim)


    # Loss
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_history = []

    comment = f'{MODEL_NAME} lr = {LR}'
    print(comment)
    tb = SummaryWriter(comment=comment)

    eval_graphDTA(model, val_data_loader, criterion, device)

    # Epoch Runs
    BEST_EPOCH, BEST_F1 = 0, -np.inf
    pbar = tqdm(range(NUM_EPOCHS), position=1)
    patience = 0
    FLAG_STOP_TRAIN = False
    SAVE_DIR = 'saved/{}/{}/{}/'.format(UPSAMPLE, EXP_TYPE,MODEL_NAME)
    BEST_MODEL_STATE = None
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    for epoch in pbar:
        pbar.set_description('Epoch: {}/{}'.format(epoch, NUM_EPOCHS))
        pbar2 = tqdm(train_data_loader, position=0, desc='Training')
        total_loss = 0
        for i, (data) in enumerate(pbar2):
            data = data.to(device)

            optim.zero_grad()

            score = model(data)
            prob = F.sigmoid(score).to('cpu').data.numpy()
            label = data.y.float().to(device)
            if len(label.shape) == 1:
                score = score.squeeze(1)
            loss = criterion(score, label)
            total_loss += loss.item()
            loss_history.append(loss.item())

            loss.backward()
            optim.step()
            lr_scheduler.step(lr_scheduler.last_epoch + 1)
            warmup_scheduler.dampen()

        tb.add_scalar("Train Loss", total_loss, epoch)
        for name, weight in model.named_parameters():
            try:
                tb.add_histogram(name, weight, epoch)
                tb.add_histogram(f'{name}.grad', weight.grad, epoch)
            except:
                # print('{} received Nonetype grad')
                continue
        if epoch % 1 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            metrics = eval_graphDTA(model, val_data_loader, criterion, device)
            val_loss = metrics['avg_loss']
            ap = metrics['avg_precision']
            auroc = metrics['auroc']
            recall = metrics['sensitivity']
            tnr = metrics['specificity']
            f1 = metrics['f1']
            tb.add_scalar("Valid Loss", val_loss, epoch)
            tb.add_scalar("Valid AUPR", ap, epoch)
            tb.add_scalar("Valid AUROC", auroc, epoch)
            tb.add_scalar("Valid F1", f1, epoch)
            if f1 > BEST_F1:
                patience = 0
                BEST_EPOCH = epoch
                BEST_F1 = f1
                SAVE_PATH = '{}/checkpoint.model'.format(SAVE_DIR)
                torch.save(model, SAVE_PATH)
                BEST_MODEL_STATE = SAVE_PATH
            else:
                patience += 1
                if patience > EPOCH_PATIENCE: FLAG_STOP_TRAIN = True
                print(f'Patience:{patience}')

            print(
                '\nBest EPOCH: {} BEST F1: {} CURRENT LOSS: {} AP: {} AUROC: {} Recall: {} TNR: {} F1: {}'.format(
                    BEST_EPOCH, BEST_F1, val_loss,
                    ap, auroc, recall, tnr, f1))

        if FLAG_STOP_TRAIN:
            break

    print('Testing...')
    model = torch.load(BEST_MODEL_STATE)
    model = model.to(device)
    metrics = eval_graphDTA(model, test_data_loader,criterion, device)
    metrics['model_state'] = BEST_MODEL_STATE
    print(metrics)

    METRICS_DIR = f'{SAVE_DIR}/metrics/'
    if not os.path.exists(METRICS_DIR): os.mkdir(METRICS_DIR)
    with open(f'{METRICS_DIR}/metrics.pkl', 'wb') as f:
        pkl.dump(metrics, f)
    end_time = time.time() - start_time
    end_time /= 60.0

    print('Test: AUROC:{} AUPR:{}'.format(metrics['auroc'], metrics['avg_precision'], metrics['f1']))

    print(f'Completed in {end_time} minutes')

def load_model():
    drug_embedding = np.load('saved/pretrain/drug/GraphSAGE/embedding.npy')
    protein_embedding = np.load('saved/pretrain/protein/GraphSAGE/embedding.npy')

    drug_embedding = torch.from_numpy(drug_embedding)
    protein_embedding = torch.from_numpy(protein_embedding)

    model = GraMDTA(num_features_xd=53, drug_hetero_embed=drug_embedding, protein_hetero_embed=protein_embedding)
    return model


exp_types = ['1v1', '1v5','1v10']
folders = ['E1']

for exp_type in exp_types:
    for folder in folders:
        print('*'*10)
        print(f'{exp_type} {folder}')
        print('*' * 10)
        train('GraMDTA', exp_type, folder)
