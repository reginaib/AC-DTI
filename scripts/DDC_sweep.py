from finetuning import start_sweep


sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/BCELoss"
        },
        "parameters": {
            "mode": {"value": "DDC"},
            "n_targets": {"value": 229},
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            # 'lr': {'max': 0.001, 'min': 0.00001},
            # 'dr': {'max': 0.5, 'min': 0.01},
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "csv": {"value": './analysis/kiba_ddc_cb_ta1_ts0.9.csv'},
            "save_preds": {"value": False},
            "checkpoint": {"value": False},
        }
}

start_sweep(config=sweep_config, project_name='DDC_KIBA_cb_sweep', num_config=100)
