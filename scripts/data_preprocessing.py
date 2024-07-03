import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from rdkit.Chem import MolFromSmiles, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from pandas import read_csv, read_parquet
from itertools import chain
from pickle import load, dump
from pytorch_lightning import LightningDataModule


class DrugDrugData(LightningDataModule):
    def __init__(self, csv=None, parquet=None, task=None, radius=2, n_bits=1024,
                 cache='cache.data', batch_size=100):
        super().__init__()
        self.prepare_data_per_node = False
        self.csv = csv
        self.parquet = parquet
        self.task = task
        self.radius = radius
        self.n_bits = n_bits
        self.cache = cache
        self.batch_size = batch_size

    def prepare_data(self):
        if self.csv is not None and self.parquet is not None:
            raise ValueError("Only one of 'csv' or 'parquet' should be provided.")
        elif self.csv is not None:
            data = read_csv(self.csv)
        elif self.parquet is not None:
            data = read_parquet(self.parquet)
        else:
            raise ValueError("Either 'csv' or 'parquet' must be provided.")

        cache = {}
        # Processing SMILES strings to build cache only for valid molecules
        for s in chain(data.smiles1, data.smiles2):
            if s not in cache:
                try:
                    mol = MolFromSmiles(s)
                    if mol is not None:
                        fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                        arr = np.zeros((0,), dtype=np.int8)
                        DataStructs.ConvertToNumpyArray(fp, arr)
                        cache[s] = torch.tensor(arr, dtype=torch.float32)
                except:
                    pass

        drugs1, drugs2, labels, splits, targets = [], [], [], [], []
        task_column = 'cliff' if self.task == 'classification' else 'affinity_difference'
        for index, row in data.iterrows():
            if row.smiles1 in cache and row.smiles2 in cache:
                drugs1.append(cache[row.smiles1])
                drugs2.append(cache[row.smiles2])
                labels.append(row[task_column])
                splits.append(row.split)
                targets.append(row.target)

        if self.csv is not None:
            # Convert lists to tensors using valid_indices to filter DataFrame directly
            drugs1 = torch.stack(drugs1)
            drugs2 = torch.stack(drugs2)
            label = torch.tensor(labels, dtype=torch.float32)
            split = torch.tensor(splits, dtype=torch.long)
            target = torch.tensor(targets, dtype=torch.long)

            # Save processed data
            with open(self.cache, 'wb') as f:
                dump((drugs1, drugs2, label, split, target), f)


        elif self.parquet is not None:
            # For Parquet, save directly as lists
            df = pd.DataFrame({
                'drugs1': [d.numpy().tolist() for d in drugs1],
                'drugs2': [d.numpy().tolist() for d in drugs2],
                'label': labels,
                'split': splits,
                'target': targets
            })

            df.to_parquet(self.cache)

    def setup(self, stage=None):
        if self.csv is not None:
            with open(self.cache, 'rb') as f:
                drugs1, drugs2, labels, split, target = load(f)

        elif self.parquet is not None:
            df = read_parquet(self.cache)

            drugs1 = torch.tensor(df['drugs1'].tolist(), dtype=torch.float32)
            drugs2 = torch.tensor(df['drugs2'].tolist(), dtype=torch.float32)
            labels = torch.tensor(df['label'].tolist(), dtype=torch.float32)
            split = torch.tensor(df['split'].tolist(), dtype=torch.long)
            target = torch.tensor(df['target'].tolist(), dtype=torch.long)

        mask = split == 0
        self._train = TensorDataset(drugs1[mask], drugs2[mask], labels[mask], target[mask])
        mask = split == 1
        self._validation = TensorDataset(drugs1[mask], drugs2[mask], labels[mask], target[mask])
        mask = split == 2
        self._test = TensorDataset(drugs1[mask], drugs2[mask], labels[mask], target[mask])

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
            if s not in cache:
                try:
                    mol = MolFromSmiles(s)
                    if mol is not None:
                        fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                        arr = np.zeros((0,), dtype=np.int8)
                        DataStructs.ConvertToNumpyArray(fp, arr)
                        cache[s] = torch.tensor(arr, dtype=torch.float32)
                except:
                    pass

        drugs, affinities, splits, targets = [], [], [], []
        for index, row in affinity.iterrows():
            if row.smiles in cache:
                drugs.append(cache[row.smiles])
                affinities.append(row.affinity)
                splits.append(row.split)
                targets.append(row.target)

        drugs = torch.stack(drugs)
        affinity = torch.tensor(affinities, dtype=torch.float32)
        split = torch.tensor(splits, dtype=torch.long)
        target = torch.tensor(targets, dtype=torch.long)
        with open(self.cache, 'wb') as f:
            dump((drugs, target, affinity, split), f)

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
