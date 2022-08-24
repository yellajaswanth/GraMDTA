import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import Chem
import re
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rdkit import DataStructs
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader.utils import to_hetero_csc
from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader.utils import filter_hetero_data

import pickle as pkl

amino_char = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
              "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
              "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
              "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

drug_kg = pkl.load(open('../data/drugbank_all_targets/drug_hetero_graph_nx.pkl', 'rb'))
g_smile2idx = {v['smiles']: v['index_'] for n, v in drug_kg.nodes(data=True) if v['type_'] == 'drug'}
g_idx2smile = {v['index_']: v['smiles'] for n, v in drug_kg.nodes(data=True) if v['type_'] == 'drug'}

protein_kg = pkl.load(open('../data/drugbank_all_targets/protein_hetero_graph_nx.pkl', 'rb'))
g_seq2idx = {v['sequence']: v['index_'] for n, v in protein_kg.nodes(data=True) if v['type_'] == 'protein'}
g_idx2seq = {v['index_']: v['sequence'] for n, v in protein_kg.nodes(data=True) if v['type_'] == 'protein'}


def label_sequence(line, sequ_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = sequ_ch_ind[ch]
    return X

class HeteroMoleculeNet(InMemoryDataset):
    '''
    Code From: https://github.com/pyg-team/pytorch_geometric/blob/a0f72f983e510690953e6073dd0f74329af07230/torch_geometric/datasets/molecule_net.py#L60
    '''

    # url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform,
                                                pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])



    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_file_names(self):
        return 'data.pt'

    @property
    def raw_file_names(self):
        return f'{self.name}.txt'

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'hetero_{self.name}_processed')

    @property
    def raw_paths(self):
        return [osp.join(self.root, self.raw_file_names)]

    def mol2tensor(self, mol):
        xs = []
        for atom in mol.GetAtoms():
            x = []
            x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
            x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
            x.append(x_map['degree'].index(atom.GetTotalDegree()))
            x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
            x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
            x.append(x_map['num_radical_electrons'].index(
                atom.GetNumRadicalElectrons()))
            x.append(x_map['hybridization'].index(
                str(atom.GetHybridization())))
            x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
            x.append(x_map['is_in_ring'].index(atom.IsInRing()))
            xs.append(x)

        x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e.append(e_map['bond_type'].index(str(bond.GetBondType())))
            e.append(e_map['stereo'].index(str(bond.GetStereo())))
            e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        return x, edge_index, edge_attr

    def process(self):

        protein_max = 1000
        data_list = []
        path = osp.join(self.root, self.name+'.txt')
        with open(path, 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]

        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split('\t')

            smiles, seq_str, y = line[-3], line[-2], line[-1]
            sequence = torch.from_numpy(label_sequence(seq_str, amino_char, protein_max))

            sequence = sequence.view(1,-1)

            y = int(float(y))
            y = torch.tensor(y).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None or sequence is None:
                continue

            x, edge_index, edge_attr = self.mol2tensor(mol)
            if smiles in g_smile2idx and seq_str in g_seq2idx:
                d_kg, p_kg = torch.tensor([g_smile2idx[smiles]]), torch.tensor([g_seq2idx[seq_str]])
            else:
                continue
                if smiles not in g_smile2idx:
                    target_idx = self.get_mol_equivalent_idx(smiles)
                    d_kg = torch.tensor([target_idx])
                if seq_str not in g_seq2idx:
                    target_idx = self.get_fasta_equivalent_idx(seq_str)
                    p_kg = torch.tensor([target_idx])



            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles, sequence=sequence, d_kg = d_kg, p_kg=p_kg)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_mol_equivalent_idx(self, smiles):
        mol1 = Chem.MolFromSmiles(smiles)
        mol1 = Chem.RDKFingerprint(mol1)
        curr_sim = 0
        target_idx = None
        for target_smile, idx in g_smile2idx.items():
            mol2 = Chem.MolFromSmiles(target_smile)
            mol2 = Chem.RDKFingerprint(mol2)
            similarity = DataStructs.FingerprintSimilarity(mol1, mol2)
            if curr_sim < similarity:
                curr_sim = similarity
                target_idx = idx
        return target_idx

    def get_fasta_equivalent_idx(self, sequence):
        mol1 = Chem.MolFromFASTA(sequence)
        mol1 = Chem.RDKFingerprint(mol1)
        curr_sim = 0
        target_idx = None
        for target_seq, idx in g_seq2idx.items():
            mol2 = Chem.MolFromFASTA(target_seq)
            mol2 = Chem.RDKFingerprint(mol2)
            similarity = DataStructs.FingerprintSimilarity(mol1, mol2)
            if curr_sim < similarity:
                curr_sim = similarity
                target_idx = idx
        return target_idx

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

class HeteroGenFeatures(object):
    def __init__(self):
        self.symbols = ['H', 'B', 'C', 'N', 'O', 'F', 'Al',
                        'Si', 'P', 'S', 'Cl', 'V', 'Fe', 'Co',
                        'Cu', 'Zn', 'As', 'Se', 'Br', 'Mo', 'Ru',
                        'Sn', 'Te', 'I', 'W', 'Au', 'Hg', 'other']

        self.prot_symbols = ['O', 'S', 'C', 'N']

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            # 'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.

        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 7
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data