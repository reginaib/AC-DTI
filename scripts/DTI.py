from finetuning import start_training


class config:
    n_hidden_layers = 3
    input_dim = 1024
    hidden_dim_d = 256
    hidden_dim_t = 256
    hidden_dim_c = 128
    lr = 1e-3
    dr = 0.1
    patience = 15
    accelerator = 'gpu'
    max_epochs = 100
    dataset_name = '../data/kiba_d_t_aff_smiles_split.csv'


start_training(mode='DDC', config=config, project_name='DTI_aff')


