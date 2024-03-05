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
from torchmetrics.classification import BinaryRecall, BinaryAccuracy, BinaryF1Score, BinaryPrecision
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import EarlyStopping


class DrugDrugCliffNN(LightningModule):
    def __init__(self, input_dim=1024, hidden_dim_d=128, hidden_dim_t=128, hidden_dim_c=128, lr=1e-4):
        super().__init__()
        self.lr = lr
        # The branch for processing each drug
        self.d_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_d, hidden_dim_d // 2),
            nn.ReLU()
        )

        self.t_encoder = nn.Embedding(230, hidden_dim_t)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_d + hidden_dim_t, hidden_dim_c),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_c, 1)
        )

        self.metrics_v = MetricCollection([BinaryRecall(), BinaryAccuracy(), BinaryF1Score(), BinaryPrecision()],
                                          prefix='validation/')

        self.metrics_t = MetricCollection([BinaryRecall(), BinaryAccuracy(), BinaryF1Score(), BinaryPrecision()],
                                          prefix='test/')
        self.save_hyperparameters()

    def forward(self, drug1, drug2, target):
        # Process each compound through the same branch
        drug1_out = self.d_encoder(drug1)
        drug2_out = self.d_encoder(drug2)
        target_out = self.t_encoder(target)

        # Concatenate the outputs
        combined_out = torch.cat((drug1_out, drug2_out, target_out), dim=1)

        # Classifier
        return self.classifier(combined_out).flatten()

    def training_step(self, batch):
        drug1, drug2, clf, target = batch

        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.log('Training/BCELoss', ls)
        return ls

    def validation_step(self, batch, _):
        drug1, drug2, clf, target = batch

        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.metrics_v.update(preds.sigmoid(), clf)
        self.log('Validation/BCELoss', ls)

    def test_step(self, batch, *_):
        drug1, drug2, clf, target = batch

        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf)
        self.metrics_t.update(preds.sigmoid(), clf)
        self.log('Test/BCELoss', ls)

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_v.compute())
        self.metrics_v.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics_t.compute())
        self.metrics_t.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


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


config = {
    "input_dim": 1024,
    "hidden_dim_d": 128,
    "hidden_dim_t": 128,
    "hidden_dim_c": 128,
    "lr": 1e-4,
    "batch_size": 128,
    "max_epochs": 10
}


wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')

model = DrugDrugCliffNN(
    input_dim=config['input_dim'],
    hidden_dim_d=config['hidden_dim_d'],
    hidden_dim_t=config['hidden_dim_t'],
    hidden_dim_c=config['hidden_dim_c'],
    lr=config['lr']
)

data = DrugDrugData('../analysis/kiba_cliff_pairs_ta1_ts0.9_cb_wt.csv')
logger = WandbLogger(project='kiba_cb', job_type='train')

early_stop_callback = EarlyStopping(
    monitor='Validation/BCELoss',
    min_delta=0.00,
    patience=5,
    verbose=True,
    mode='min'
)

trainer = Trainer(
    accelerator='cpu',
    max_epochs=config['max_epochs'],
    logger=logger,
    callbacks=[early_stop_callback]  # Include the callback here
)

trainer.fit(model, data)
trainer.test(model, data)
