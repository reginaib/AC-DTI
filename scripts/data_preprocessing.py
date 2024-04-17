import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rdkit.Chem import MolFromSmiles, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from pandas import read_csv
from itertools import chain
from pickle import load, dump
from pytorch_lightning import LightningDataModule


class DrugDrugData(LightningDataModule):
    def __init__(self, csv, radius=2, n_bits=1024, cache='cache.data', batch_size=100):
        super().__init__()
        self.prepare_data_per_node = False
        self.csv = csv
        self.radius = radius
        self.n_bits = n_bits
        self.cache = cache
        self.batch_size = batch_size

    def prepare_data(self):
        # expected next columns: smiles1, smiles2, cliff, split, target
        data = read_csv(self.csv)

        cache = {}
        for s in chain(data.smiles1, data.smiles2):
            if s in cache:
                continue
            mol = MolFromSmiles(s)
            fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            cache[s] = torch.tensor(arr, dtype=torch.float32)

        drugs1, drugs2 = [], []
        for row in data.itertuples():
            drugs1.append(cache[row.smiles1])
            drugs2.append(cache[row.smiles2])
        drugs1 = torch.stack(drugs1)
        drugs2 = torch.stack(drugs2)
        cliff = torch.tensor(data.cliff, dtype=torch.float32)
        split = torch.tensor(data.split, dtype=torch.long)
        target = torch.tensor(data.target, dtype=torch.long)
        with open(self.cache, 'wb') as f:
            dump((drugs1, drugs2, cliff, split, target), f)

    def setup(self, stage=None):
        with open(self.cache, 'rb') as f:
            drugs1, drugs2, cliff, split, target = load(f)

        mask = split == 0
        self._train = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask], target[mask])
        mask = split == 1
        self._validation = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask], target[mask])
        mask = split == 2
        self._test = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask], target[mask])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size)


class DrugTargetData(LightningDataModule):
    def __init__(self, affinity, radius=2, n_bits=1024, cache='cache.data', batch_size=100):
        super().__init__()
        self.prepare_data_per_node = False
        self.affinity = affinity
        self.radius = radius
        self.n_bits = n_bits
        self.cache = cache
        self.batch_size = batch_size

    def prepare_data(self):
        # : smiles, target, affinity, split
        affinity = read_csv(self.affinity)

        cache = {}
        for s in chain(affinity.smiles):
            if s in cache:
                continue
            mol = MolFromSmiles(s)
            fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            cache[s] = torch.tensor(arr, dtype=torch.float32)

        drugs = torch.stack([cache[x] for x in affinity.smiles])
        target1 = torch.tensor(affinity.target, dtype=torch.long)
        split = torch.tensor(affinity.split, dtype=torch.long)
        affinity = torch.tensor(affinity.affinity, dtype=torch.float32)
        with open(self.cache, 'wb') as f:
            dump((drugs, target1, affinity, split), f)

    def setup(self, stage=None):
        with open(self.cache, 'rb') as f:
            drugs, target1, affinity, split = load(f)

        mask = split == 0
        self._train = TensorDataset(drugs[mask], target1[mask], affinity[mask])
        mask = split == 1
        self._validation = TensorDataset(drugs[mask], target1[mask], affinity[mask])
        mask = split == 2
        self._test = TensorDataset(drugs[mask], target1[mask], affinity[mask])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size)