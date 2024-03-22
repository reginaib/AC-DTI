from finetuning import start_sweep


sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/MAELoss"
        },
        "parameters": {
            "mode": {"value": "DTI"},
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 756, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 756, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 756, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            # 'lr': {'max': 0.001, 'min': 0.00001},
            # 'dr': {'max': 0.5, 'min': 0.01},
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "dataset_name": {"value": 'data/kiba_d_t_aff_smiles_split.csv'},
        }
}


start_sweep(config=sweep_config, project_name='DTI_aff_sweep', num_config=50)
