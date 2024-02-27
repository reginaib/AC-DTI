import numpy as np
import torch
import wandb
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from rdkit.Chem import MolFromSmiles, DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from pandas import read_csv
from itertools import chain
from pickle import load, dump


class DrugDrugCliffNN(LightningModule):
    def __init__(self, input_dim=1024, hidden_dim=128):
        super().__init__()
        # The branch for processing each compound
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # The classifier part that operates on the concatenated output of compound branches
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, drug1, drug2):
        # Process each compound through the same branch
        drug1_out = self.encoder(drug1)
        drug2_out = self.encoder(drug2)

        # Concatenate the outputs
        combined_out = torch.cat((drug1_out, drug2_out), dim=1)

        # Classifier
        return self.classifier(combined_out).flatten()

    def training_step(self, batch):
        drug1, drug2, clf = batch

        preds = self(drug1, drug2)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.log('Training/BCELoss', ls)
        return ls

    def validation_step(self, batch, _):
        drug1, drug2, clf = batch

        preds = self(drug1, drug2)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.log('Validation/BCELoss', ls)

    def test_step(self, batch, *_):
        drug1, drug2, clf = batch

        preds = self(drug1, drug2)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.log('Test/BCELoss', ls)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


class DrugDrugData(LightningDataModule):
    def __init__(self, csv, radius=2, n_bits=1024, cache='cache.data', batch_size=10):
        super().__init__()
        self.prepare_data_per_node = False
        self.csv = csv
        self.radius = radius
        self.n_bits = n_bits
        self.cache = cache
        self.batch_size = batch_size

    def prepare_data(self):
        # expected next columns: smiles1, smiles2, cliff, split
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
        split = torch.tensor(data.split, dtype=torch.int8)
        with open(self.cache, 'wb') as f:
            dump((drugs1, drugs2, cliff, split), f)

    def setup(self, stage=None):
        with open(self.cache, 'rb') as f:
            drugs1, drugs2, cliff, split = load(f)

        mask = split == 0
        self._train = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask])
        mask = split == 1
        self._validation = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask])
        mask = split == 2
        self._test = TensorDataset(drugs1[mask], drugs2[mask], cliff[mask])

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size)


wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
model = DrugDrugCliffNN()
data = DrugDrugData('../data/KIBA/kiba_cliff_pairs_ta_1_ts_0.9_cb.csv')
logger = WandbLogger(project='kiba_cb', job_type='train')

trainer = Trainer(accelerator='cpu', max_epochs=10, logger=logger)
trainer.fit(model, data)
trainer.test(model, data)
