import torch
import wandb

from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.classification import (BinaryRecall, BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryAUROC, BinaryMatthewsCorrCoef)
from torchmetrics import MetricCollection
from torcheval.metrics import BinaryAUPRC
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import LightningModule, Trainer
from data_preprocessing import DrugDrugData


class DrugDrugCliffNN(LightningModule):
    def __init__(self, n_hidden_layers, hidden_dim_d, hidden_dim_t, hidden_dim_c, lr, dr,
                 input_dim=1024, n_targets=222, pos_weight=2, min_hidden_dim=32):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        # Calculate hidden dimensions, ensuring they do not go below min_hidden_dim
        hidden_dims_d = [max(hidden_dim_d // (2 ** i), min_hidden_dim) for i in range(n_hidden_layers)]

        # The branch for processing each drug
        layers = [nn.Linear(input_dim, hidden_dims_d[0]), nn.ReLU(), nn.Dropout(dr)]
        for i in range(1, n_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dims_d[i - 1], hidden_dims_d[i]),
                nn.ReLU(),
                nn.Dropout(dr)
            ])
        self.d_encoder = nn.Sequential(*layers)

        self.t_encoder = nn.Embedding(n_targets, hidden_dim_t)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims_d[-1] * 2 + hidden_dim_t, hidden_dim_c),
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
                                           prefix='train/')
        self.metric_prc_tr = BinaryAUPRC()

        self.metrics_v = MetricCollection([BinaryRecall(),
                                           BinaryAccuracy(),
                                           BinaryF1Score(),
                                           BinaryPrecision(),
                                           BinaryAUROC(),
                                           BinaryMatthewsCorrCoef()],
                                          prefix='validation/')
        self.metric_prc_v = BinaryAUPRC()

        self.metrics_t = MetricCollection([BinaryRecall(),
                                           BinaryAccuracy(),
                                           BinaryF1Score(),
                                           BinaryPrecision(),
                                           BinaryAUROC(),
                                           BinaryMatthewsCorrCoef()],
                                          prefix='test/')
        self.metric_prc_t = BinaryAUPRC()
        self.save_hyperparameters()

        # To make sure it works
        print(self)


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
        self.metric_prc_tr.reset()
        self.metrics_tr.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.metrics_v.compute())
        self.log('Validation/BinaryAUPRC', self.metric_prc_v.compute())
        self.metric_prc_v.reset()
        self.metrics_v.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.metrics_t.compute())
        self.log('Test/BinaryAUPRC', self.metric_prc_t.compute())
        self.metric_prc_t.reset()
        self.metrics_t.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def optimize():
    wandb.init()
    config = wandb.config
    logger = WandbLogger()
    model = DrugDrugCliffNN(n_hidden_layers=config.n_hidden_layers,
                            hidden_dim_d=config.hidden_dim_d,
                            hidden_dim_t=config.hidden_dim_t,
                            hidden_dim_c=config.hidden_dim_c,
                            lr=config.lr,
                            dr=config.dr)

    early_stop_callback = EarlyStopping(
        monitor='Validation/BCELoss',
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        accelerator='gpu',
        max_epochs=50,
        logger=logger,
        callbacks=[early_stop_callback]
    )

    data = DrugDrugData('../analysis/kiba_cliff_pairs_ta1_ts0.9_r_wt.csv')

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
        'hidden_dim_d': {'values': [32, 64, 128, 256, 512, 768, 1024]},
        'hidden_dim_t': {'values': [32, 64, 128, 256, 512, 768, 1024]},
        'hidden_dim_c': {'values': [32, 64, 128, 256, 512, 768, 1024]},
        'lr': {'values': [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
        'dr': {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
    }
}

wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
sweep_id = wandb.sweep(sweep_config, project='DDC_sweep_r_bn')
wandb.agent(sweep_id=sweep_id, function=optimize, count=15)
