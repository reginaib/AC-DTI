from finetuning import start_sweep

# Define the configuration for the hyperparameter sweep
sweep_config = {
        "method": "random",
        "metric": {
            "goal": "minimize",
            "name": "Validation/BCELoss"
        },
        "parameters": {
            # Fixed parameters for the sweep
            "mode": {"value": "DDC"},

            # Number of unique targets in the dataset
            "n_targets": {"value": 229},

            # Parameters with multiple values to test during the sweep
            "n_hidden_layers": {"values": [1, 2, 3, 4]},
            "hidden_dim_d": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_t": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "hidden_dim_c": {"values": [32, 64, 128, 256, 512, 768, 1024]},
            "lr": {"values": [0.00001, 0.00003, 0.0001, 0.0003, 0.001]},
            "dr": {'values': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},

            # Additional fixed parameters for the sweep
            "patience": {"value": 15},
            "accelerator": {"value": "gpu"},
            "max_epochs": {"value": 100},
            "csv": {"value": './analysis/kiba_ddc_cb_ta1_ts0.9.csv'},
            "save_preds": {"value": False},
            "checkpoint": {"value": False},
        }
}

# Start the hyperparameter sweep with the defined configuration
# Arguments:
# - config: The configuration dictionary for the sweep
# - project_name: The name of the project in Weights and Biases where the sweep results will be logged
# - num_config: The number of different configurations to test during the sweep
start_sweep(config=sweep_config, project_name='DDC_KIBA_cb_sweep', num_config=100)

