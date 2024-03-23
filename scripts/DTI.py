from finetuning import start_training


class config:
    n_hidden_layers = 1
    hidden_dim_d = 1024
    hidden_dim_t = 756
    hidden_dim_c = 756
    lr = 0.0003
    dr = 0.01
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = './analysis/kiba_d_t_aff_smiles_split.csv'
    pre_trained_d_encoder_path = None


start_training(mode='DTI', config=config, project_name='DTI_aff_train')


