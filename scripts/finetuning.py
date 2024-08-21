import wandb
import torch

from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from models import DrugDrugCliffNN, DrugTargetAffNN
from data_preprocessing import DrugDrugData, DrugTargetData


def initialize_model(mode, config, logger):
    """
    Initialize the model, data, and trainer, and start the training and testing process.

    Args:
        mode (str): The mode of operation, either 'DDC' (Drug-Drug Cliff) or 'DTI' (Drug-Target Interaction).
        config: Configuration object containing hyperparameters and other settings.
        logger: Logger for tracking experiment metrics and progress (e.g., WandbLogger).
    """

    # Initialize the appropriate model based on the mode
    if mode == 'DDC':
        model = DrugDrugCliffNN(n_hidden_layers=config.n_hidden_layers,
                                hidden_dim_d=config.hidden_dim_d,
                                hidden_dim_t=config.hidden_dim_t,
                                hidden_dim_c=config.hidden_dim_c,
                                lr=config.lr,
                                dr=config.dr,
                                n_targets=config.n_targets)

        monitor = 'Validation/BCELoss'

    elif mode == 'DTI':
        model = DrugTargetAffNN(n_hidden_layers=config.n_hidden_layers,
                                hidden_dim_d=config.hidden_dim_d,
                                hidden_dim_t=config.hidden_dim_t,
                                hidden_dim_c=config.hidden_dim_c,
                                lr=config.lr,
                                dr=config.dr,
                                n_targets=config.n_targets,
                                pre_trained_d_encoder_path=config.pre_trained_d_encoder_path,
                                pre_trained_t_encoder_path=config.pre_trained_t_encoder_path,
                                freeze=config.freeze,
                                layer_to_d_encoder=config.layer_to_d_encoder,
                                hidden_dim_d_add=config.hidden_dim_d_add,
                                dr2=config.dr2)
        monitor = 'Validation/MAELoss'

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=config.patience,
        verbose=True,
        mode='min')

    # Set up checkpoint callback to save the model during training
    checkpoint_callback = ModelCheckpoint(filename='./results/{epoch:02d}')

    callbacks = [early_stop_callback]

    # Add checkpoint callback if specified in config
    if config.checkpoint:
        callbacks.append(checkpoint_callback)

    trainer = Trainer(accelerator=config.accelerator,
                      max_epochs=config.max_epochs,
                      logger=logger,
                      callbacks=callbacks)

    # Initialize the data module based on the mode
    if mode == 'DDC':
        data = DrugDrugData(csv=config.csv)
    elif mode == 'DTI':
        data = DrugTargetData(config.dataset_name)

    trainer.fit(model, data)
    trainer.test(model, data)

    # Save the predictions if specified in the config
    if config.save_preds:
        predictions = torch.cat(trainer.predict(model, data.test_dataloader()))
        torch.save(predictions, f'./results/{config.preds_name}')


def optimize_sweep():
    """
    Initialize and optimize the model based on the hyperparameters provided by Weights and Biases sweep.
    """

    wandb.init()
    config = wandb.config
    logger = WandbLogger()
    initialize_model(config.mode, config=config, logger=logger)


def start_sweep(config, project_name, num_config=15):
    """
    Start a hyperparameter sweep using Weights and Biases.

    Args:
        config: The sweep configuration.
        project_name (str): The name of the Weights and Biases project.
        num_config (int, optional): Number of configurations to run during the sweep. Defaults to 15.
    """

    wandb.login(key='...')
    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=optimize_sweep, count=num_config)


def start_training(mode, config, project_name):
    """
    Start the training process for a given mode and configuration, with Weights and Biases logging.

    Args:
        mode (str): The mode of operation, either 'DDC' or 'DTI'.
        config: Configuration object containing hyperparameters and other settings.
        project_name (str): The name of the Weights and Biases project.
    """

    wandb.login(key='...')
    logger = WandbLogger(project=project_name, job_type='train', log_model='all')
    initialize_model(mode, config, logger)
