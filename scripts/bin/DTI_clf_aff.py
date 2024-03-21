import numpy as np
import torch
import wandb
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
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.utilities import CombinedLoader
from pytorch_lightning import LightningModule, Trainer, LightningDataModule


class DrugDrugCliffNN(LightningModule):
    def __init__(self, n_hidden_layers=2, input_dim=1024, hidden_dim_d=128, hidden_dim_t=128, hidden_dim_c=128,
                 lr=1e-4, dr=0.1, n_targets=229, alpha=1.):
        super().__init__()
        self.lr = lr
        self.alpha = alpha

        # The branch for processing each drug
        layers = [nn.Linear(input_dim, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim_d, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)])
        self.d_encoder = nn.Sequential(*layers)

        self.t_encoder = nn.Embedding(n_targets, hidden_dim_t)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim_d + hidden_dim_t, hidden_dim_c),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim_c, 1)
        )

        self.metrics_tr = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='train/')

        self.metrics_v = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='validation/')

        self.metrics_t = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='test/')

        self.save_hyperparameters()

    def forward(self, drug, target):
        # Process each compound through the same branch
        drug_out = self.d_encoder(drug)
        target_out = self.t_encoder(target)

        combined_out = torch.cat((drug_out, target_out), dim=1)

        return self.regressor(combined_out).flatten()

    def training_step(self, batch):
        data_lists, _, _ = batch

        aff_batch, clf_batch = data_lists
        drug, target1, aff = aff_batch[0], aff_batch[1], aff_batch[2]
        drug1, drug2, target2, clf = clf_batch[0], clf_batch[1], clf_batch[2], clf_batch[3]

        preds = self(drug, target1)
        ls = F.l1_loss(preds, aff)
        self.metrics_tr.update(preds, aff)
        self.log('Training/MAELoss', ls)

        drug1_out = self.d_encoder(drug1)
        drug2_out = self.d_encoder(drug2)
        target2_out = self.t_encoder(target2)

        combined1_out = torch.cat((drug1_out, target2_out), dim=1)
        combined2_out = torch.cat((drug2_out, target2_out), dim=1)
        hidden1 = self.regressor[:-1](combined1_out).detach()
        hidden2 = self.regressor[:-1](combined2_out)
        contrast = F.cosine_similarity(hidden1, hidden2) * clf + 1
        contrast = contrast.mean()
        self.log('Training/CosineLoss', contrast)
        return ls + self.alpha * contrast

    def validation_step(self, batch, _):
        drug, target, aff = batch

        preds = self(drug, target)
        ls = F.l1_loss(preds, aff)
        self.metrics_v.update(preds, aff)
        self.log('Validation/MAELoss', ls)

    def test_step(self, batch, *_):
        drug, target, aff = batch

        preds = self(drug, target)
        ls = F.l1_loss(preds, aff)
        self.metrics_t.update(preds, aff)
        self.log('Test/MAELoss', ls)

    def on_train_epoch_end(self):
        self.log_dict(self.metrics_tr.compute())
        self.metrics_tr.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_v.compute())
        self.metrics_v.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics_t.compute())
        self.metrics_t.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class DrugDrugData(LightningDataModule):
    def __init__(self, affinity, cliff, radius=2, n_bits=1024, cache='cache.data', batch_size=100):
        super().__init__()
        self.prepare_data_per_node = False
        self.affinity = affinity
        self.cliff = cliff
        self.radius = radius
        self.n_bits = n_bits
        self.cache = cache
        self.batch_size = batch_size

    def prepare_data(self):
        # expected next columns: smiles1, smiles2, cliff, target
        cliff = read_csv(self.cliff)
        # : smiles, target, affinity, split
        affinity = read_csv(self.affinity)

        cache = {}
        for s in chain(cliff.smiles1, cliff.smiles2, affinity.smiles):
            if s in cache:
                continue
            mol = MolFromSmiles(s)
            fp = GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((0,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            cache[s] = torch.tensor(arr, dtype=torch.float32)

        drugs1, drugs2 = [], []
        for row in cliff.itertuples():
            drugs1.append(cache[row.smiles1])
            drugs2.append(cache[row.smiles2])
        drugs1 = torch.stack(drugs1)
        drugs2 = torch.stack(drugs2)
        target2 = torch.tensor(cliff.target, dtype=torch.long)
        cliff = torch.tensor(cliff.cliff, dtype=torch.float32)

        drugs = torch.stack([cache[x] for x in affinity.smiles])
        target1 = torch.tensor(affinity.target, dtype=torch.long)
        split = torch.tensor(affinity.split, dtype=torch.long)
        affinity = torch.tensor(affinity.affinity, dtype=torch.float32)
        with open(self.cache, 'wb') as f:
            dump((drugs, target1, affinity, split, drugs1, drugs2, target2, cliff), f)

    def setup(self, stage=None):
        with open(self.cache, 'rb') as f:
            drugs, target1, affinity, split, drugs1, drugs2, target2, cliff = load(f)

        mask = split == 0
        self._train = TensorDataset(drugs[mask], target1[mask], affinity[mask])
        mask = split == 1
        self._validation = TensorDataset(drugs[mask], target1[mask], affinity[mask])
        mask = split == 2
        self._test = TensorDataset(drugs[mask], target1[mask], affinity[mask])

        self._cliff = TensorDataset(drugs1, drugs2, target2, cliff)

    def train_dataloader(self):
        aff = DataLoader(self._train, batch_size=self.batch_size, shuffle=True)
        clf = DataLoader(self._cliff, batch_size=self.batch_size, shuffle=True)
        return CombinedLoader([aff, clf], mode='max_size_cycle')

    def val_dataloader(self):
        return DataLoader(self._validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size)


config = {
    "n_hidden_layers": 3,
    "input_dim": 1024,
    "hidden_dim_d": 256,
    "hidden_dim_t": 256,
    "hidden_dim_c": 128,
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 30,
    "dr": 0.1,
}

wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')

model = DrugDrugCliffNN(
    n_hidden_layers=config['n_hidden_layers'],
    input_dim=config['input_dim'],
    hidden_dim_d=config['hidden_dim_d'],
    hidden_dim_t=config['hidden_dim_t'],
    hidden_dim_c=config['hidden_dim_c'],
    lr=config['lr'],
    dr=config['dr']
)

data = DrugDrugData('../analysis/kiba_d_t_aff_smiles_split.csv', '../analysis/kiba_cliff_pairs_ta1_ts0.9_wt.csv')
logger = WandbLogger(project='DTI_aff_clf', job_type='train')

early_stop_callback = EarlyStopping(
    monitor='Validation/MAELoss',
    min_delta=0.00,
    patience=5,
    verbose=True,
    mode='min'
)

trainer = Trainer(
    accelerator='gpu',
    max_epochs=config['max_epochs'],
    logger=logger,
    callbacks=[early_stop_callback]
)

trainer.fit(model, data)
trainer.test(model, data)


