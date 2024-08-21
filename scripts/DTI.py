from finetuning import start_training


# Define a configuration class with various hyperparameters and settings for training
class config:
    n_hidden_layers = 1
    n_targets = 676
    hidden_dim_d = 256
    hidden_dim_t = 256
    hidden_dim_c = 1024
    lr = 0.0001
    dr = 0.01
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_d_t_aff_smiles_split.csv'
    # pre_trained_d_encoder_path = './results/DDC_KIBA_best_epoch_23.ckpt'
    pre_trained_d_encoder_path = None
    preds_name = 'DTI_vary_all_train_best_1404'
    freeze = False
    layer_to_d_encoder = False
    hidden_dim_d_add = 0
    dr2 = 0
    save_preds = True
    checkpoint = False


# Start the training process with the specified mode and configuration
# Arguments:
# - mode: The mode of operation ('DTI')
# - config: The configuration class containing all the hyperparameters and settings
# - project_name: The name of the project in Weights and Biases where the training results will be logged
start_training(mode='DTI', config=config, project_name='DTI_vary_all_best_train')
