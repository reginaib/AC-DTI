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
from torchmetrics.classification import BinaryRecall, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryAUROC, BinaryMatthewsCorrCoef
from torchmetrics import MetricCollection
from torcheval.metrics import BinaryAUPRC
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import LightningModule, Trainer, LightningDataModule


class DrugDrugCliffNN(LightningModule):
    def __init__(self, n_hidden_layers=2, input_dim=1024, hidden_dim_d=128, hidden_dim_t=128, hidden_dim_c=128,
                 lr=1e-4, dr=0.1, n_targets=222, pos_weight=6):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        # adjusting dims
        hidden_dim_t = int(hidden_dim_d * 0.5)
        hidden_dim_c = int(hidden_dim_t * 0.5)

        # The branch for processing each drug
        layers = [nn.Linear(input_dim, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim_d, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)])
        self.d_encoder = nn.Sequential(*layers)

        self.t_encoder = nn.Embedding(n_targets, hidden_dim_t)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_d + hidden_dim_d + hidden_dim_t, hidden_dim_c),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_dim_c, 1)
        )

        self.metrics_tr = MetricCollection([BinaryRecall(),
                                            BinaryAccuracy(),
                                            BinaryF1Score(),
                                            BinaryPrecision(),
                                            BinaryAUROC(),
                                            BinaryMatthewsCorrCoef()],
                                           prefix='Train/')
        self.metric_prc_tr = BinaryAUPRC()

        self.metrics_v = MetricCollection([BinaryRecall(),
                                           BinaryAccuracy(),
                                           BinaryF1Score(),
                                           BinaryPrecision(),
                                           BinaryAUROC(),
                                           BinaryMatthewsCorrCoef()],
                                          prefix='Validation/')
        self.metric_prc_v = BinaryAUPRC()

        self.metrics_t = MetricCollection([BinaryRecall(),
                                           BinaryAccuracy(),
                                           BinaryF1Score(),
                                           BinaryPrecision(),
                                           BinaryAUROC(),
                                           BinaryMatthewsCorrCoef()],
                                          prefix='Test/')
        self.metric_prc_t = BinaryAUPRC()
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
        ls = F.binary_cross_entropy_with_logits(preds, clf,
                                                pos_weight=torch.tensor(self.pos_weight, device=self.device))
        self.metrics_tr.update(preds.sigmoid(), clf.long())
        self.metric_prc_tr.update(preds.sigmoid(), clf.long())
        self.log('Training/BCELoss', ls)
        return ls

    def validation_step(self, batch, _):
        drug1, drug2, clf, target = batch

        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf,
                                                pos_weight=torch.tensor(self.pos_weight, device=self.device))
        self.metrics_v.update(preds.sigmoid(), clf.long())
        self.metric_prc_v.update(preds.sigmoid(), clf.long())
        self.log('Validation/BCELoss', ls)

    def test_step(self, batch, *_):
        drug1, drug2, clf, target = batch

        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf,
                                                pos_weight=torch.tensor(self.pos_weight, device=self.device))
        self.metrics_t.update(preds.sigmoid(), clf.long())
        self.metric_prc_t.update(preds.sigmoid(), clf.long())
        self.log('Test/BCELoss', ls)

    def on_train_epoch_end(self):
        self.log_dict(self.metrics_tr.compute())
        self.log('Train/BinaryAUPRC', self.metric_prc_tr.compute())
        self.metric_prc_v.reset()
        self.metrics_tr.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_v.compute())
        self.log('Validation/BinaryAUPRC', self.metric_prc_v.compute())
        self.metric_prc_v.reset()
        self.metrics_v.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics_t.compute())
        self.log('Test/BinaryAUPRC', self.metric_prc_t.compute())
        self.metric_prc_v.reset()
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


def optimize():
    wandb.init()
    config = wandb.config
    logger = WandbLogger()
    model = DrugDrugCliffNN(hidden_dim_d=config.hidden_dim_d, lr=config.lr, dr=config.dr)

    early_stop_callback = EarlyStopping(
        monitor='Validation/BCELoss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=30,
        logger=logger,
        callbacks=[early_stop_callback]
    )

    data = DrugDrugData('data/kiba_cliff_pairs_ta1_ts0.9_r_wt.csv')

    trainer.fit(model, data)
    trainer.test(model, data)


sweep_config = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'Validation/BCELoss'
    },
    'parameters': {
        'n_hidden_layers': {'values': [1, 2, 3, 4]},
        'hidden_dim_d': {'values': [128, 256, 512, 756, 1024]},
        'lr': {'max': 0.001, 'min': 0.00001},
        'dr': {'max': 0.5, 'min': 0.01},
    }
}

wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
sweep_id = wandb.sweep(sweep_config, project='sweep_r_bn')
wandb.agent(sweep_id=sweep_id, function=optimize, count=10)
