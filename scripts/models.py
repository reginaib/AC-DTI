import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics.classification import (BinaryRecall, BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryAUROC,
                                         BinaryMatthewsCorrCoef)
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError, R2Score
from torcheval.metrics import BinaryAUPRC
from pytorch_lightning import LightningModule


class DrugDrugCliffNN(LightningModule):
    def __init__(self, n_hidden_layers, hidden_dim_d, hidden_dim_t, hidden_dim_c, lr, dr, n_targets,
                 input_dim=1024, pos_weight=2):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight

        # Encoder for the drug representations
        layers = [nn.Linear(input_dim, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim_d, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)])
        self.d_encoder = nn.Sequential(*layers)

        # Embedding layer for encoding target information
        self.t_encoder = nn.Embedding(n_targets, hidden_dim_t)

        # Classifier that combines the outputs of the drug encoders and target encoder
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
        # Process each compound through its encoder and concatenate the outputs
        drug1_out = self.d_encoder(drug1)
        drug2_out = self.d_encoder(drug2)
        target_out = self.t_encoder(target)
        combined_out = torch.cat((drug1_out, drug2_out, target_out), dim=1)

        # Classifier prediction
        return self.classifier(combined_out).flatten()

    def training_step(self, batch):
        drug1, drug2, clf, target = batch
        preds = self(drug1, drug2, target)
        ls = F.binary_cross_entropy_with_logits(preds, clf,
                                                pos_weight=torch.tensor(self.pos_weight, device=self.device))
        self.metrics_tr.update(preds.sigmoid(), clf.long())
        self.metric_prc_tr.update(preds.sigmoid(), clf.long())
        self.log('Train/BCELoss', ls)
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


class DrugTargetAffNN(LightningModule):
    def __init__(self, n_hidden_layers, hidden_dim_d, hidden_dim_t, hidden_dim_c, lr, dr, n_targets,
                 input_dim=1024, pre_trained_d_encoder_path=None, freeze=False,
                 layer_to_d_encoder=False, hidden_dim_d_add=0, dr2=0):
        super().__init__()
        self.lr = lr

        # Drug encoder
        layers = [nn.Linear(input_dim, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim_d, hidden_dim_d), nn.ReLU(), nn.Dropout(dr)])
        self.d_encoder = nn.Sequential(*layers)

        # Load a pre-trained d_encoder model if the path is provided.
        if pre_trained_d_encoder_path:
            w = torch.load(pre_trained_d_encoder_path)
            self.load_state_dict(
                {k: v for k, v in w['state_dict'].items() if k.startswith('d_encoder')}, strict=False)

        if freeze:
            for param in self.d_encoder.parameters():
                param.requires_grad = False

        if layer_to_d_encoder:
            self.d_encoder.extend([nn.Linear(hidden_dim_d, hidden_dim_d_add), nn.ReLU(), nn.Dropout(dr2)])

        # Target encoder
        self.t_encoder = nn.Embedding(n_targets, hidden_dim_t)

        # Regression layer:
        if layer_to_d_encoder:
            regressor_layers = [nn.Linear(hidden_dim_d_add + hidden_dim_t, hidden_dim_c),
                                nn.ReLU(),
                                nn.Dropout(dr)]
        else:
            regressor_layers = [nn.Linear(hidden_dim_d + hidden_dim_t, hidden_dim_c),
                                nn.ReLU(),
                                nn.Dropout(dr)]

        regressor_layers.append(nn.Linear(hidden_dim_c, 1))
        self.regressor = nn.Sequential(*regressor_layers)

        self.metrics_tr = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='Train/')

        self.metrics_v = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='Validation/')

        self.metrics_t = MetricCollection({
            'RMSE': MeanSquaredError(squared=False),
            'MSE': MeanSquaredError(),
            'MAE': MeanAbsoluteError(),
            'R2': R2Score(),
        }, prefix='Test/')

        self.save_hyperparameters()

    def forward(self, drug, target):
        drug_out = self.d_encoder(drug)
        target_out = self.t_encoder(target)
        combined_out = torch.cat((drug_out, target_out), dim=1)
        return self.regressor(combined_out).flatten()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        drug, target, aff = batch
        return self(drug, target)

    def training_step(self, batch):
        drug, target1, aff = batch

        preds = self(drug, target1)
        ls = F.l1_loss(preds, aff)
        self.metrics_tr.update(preds, aff)
        self.log('Train/MAELoss', ls)
        return ls

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
