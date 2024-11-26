from finetuning import start_training


# Define a configuration class with various hyperparameters and settings for training
class config:
    n_hidden_layers = 1
    n_targets = 229
    hidden_dim_d = 768
    hidden_dim_t = 768
    hidden_dim_c = 512
    lr = 0.001
    dr = 0.1
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_dti_cb_split.csv'
    # pre_trained_d_encoder_path = './results/DDC_best_epoch_23.ckpt'
    pre_trained_d_encoder_path = None
    # pre_trained_t_encoder_path = './results/DDC_best_epoch_23.ckpt'
    pre_trained_t_encoder_path = None
    preds_name = 'DTI_KIBA_cb_bl_best_preds_3'
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
start_training(mode='DTI', config=config, project_name='DTI_KIBA_cb_bl_best_train')
