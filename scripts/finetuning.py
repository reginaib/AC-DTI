import wandb
import torch

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from models import DrugDrugCliffNN, DrugTargetAffNN
from data_preprocessing import DrugDrugData, DrugTargetData


def initialize_model(mode, config, logger):
    if mode == 'DDC':
        model = DrugDrugCliffNN(n_hidden_layers=config.n_hidden_layers,
                                hidden_dim_d=config.hidden_dim_d,
                                hidden_dim_t=config.hidden_dim_t,
                                hidden_dim_c=config.hidden_dim_c,
                                lr=config.lr,
                                dr=config.dr)
        monitor = 'Validation/BCELoss'
    elif mode == 'DTI':
        model = DrugTargetAffNN(n_hidden_layers=config.n_hidden_layers,
                                hidden_dim_d=config.hidden_dim_d,
                                hidden_dim_t=config.hidden_dim_t,
                                hidden_dim_c=config.hidden_dim_c,
                                lr=config.lr,
                                dr=config.dr,
                                pre_trained_d_encoder_path=config.pre_trained_d_encoder_path,
                                freeze=config.freeze)
        monitor = 'Validation/MAELoss'

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=config.patience,
        verbose=True,
        mode='min')

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}')

    trainer = Trainer(
        accelerator=config.accelerator,
        max_epochs=config.max_epochs,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback])

    if mode == 'DDC':
        data = DrugDrugData(config.dataset_name)
    elif mode == 'DTI':
        data = DrugTargetData(config.dataset_name)

    trainer.fit(model, data)
    trainer.test(model, data)

    if mode == 'DTI':
        predictions = torch.cat(trainer.predict(model, data.test_dataloader()))
        torch.save(predictions, f'./results/{config.preds_name}')


def optimize_sweep():
    wandb.init()
    config = wandb.config
    logger = WandbLogger(log_model='all')
    initialize_model(config.mode, config=config, logger=logger)


def start_sweep(config, project_name, num_config=15):
    wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=optimize_sweep, count=num_config)


def start_training(mode, config, project_name):
    wandb.login(key='fd8f6e44f8d81be3a652dbd8f4a47a7edf59e44c')
    logger = WandbLogger(project=project_name, job_type='train', log_model='all')
    initialize_model(mode, config, logger)
