# Working scripts 
## Preprocessing Notebooks
* `ds_kiba_preprocess.ipynb`:  Notebook for preprocessing the KIBA dataset.
* `ds_bindingdb_preprocess.ipynb`: Notebook for preprocessing the BindingDB dataset.

## Sweep and Training Scripts
* `DDC_sweep.py`: Script to run parameter sweeps for the DDC task.
* `DTI_sweep.py` Script to run parameter sweeps for the DTI baseline and transfer learning task.
* `DDC_sweep_bn.py`: Script to analyze the effect of the bottleneck on the DDC task.


* `DDC_wt.py`:  Script to train the DDC model with specified parameters.
* `DTI.py`: Script to train the DTI baseline and transfer learning model with specified parameters.

## Postprocessing Notebooks
* `ds_kiba_postprocess.ipynb`: Notebook for postprocessing the results of the DTI baseline and transfer learning tasks on the KIBA dataset.
* `ds_bindingdb_postprocess.ipynb`: Notebook for postprocessing the results of the DTI baseline and transfer learning tasks on the BindingDB dataset.

# Scripts with helper functions
## Data Handling
* `cliffs.py`: Contains functions for comparing chemical structures represented by SMILES strings.
* `dataset.py`: Contains functions for obtaining cliffs, splitting data, and preprocessing datasets during download.
* `data_prepocessing.py`: Contains functions for further preprocessing the data before training.

## Model Initialization and Training
* `models.py`: Contains methods for initializing models.
* `finetuning.py`: Contains methods for fine-tuning models.

## Visualization and Metrics
* `graphics.py`: Contains functions for graphs creating
* `metrics.py`: Contains functions to access the performance of the model




