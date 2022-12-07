#The original Temporal Fusion Transformers is from https://github.com/google-research/google-research/tree/master/tft

## Code Information
This repository contains the source code for the IceTFT which based on Temporal Fusion Transformer. We modified the design of input and the loss function.
The known future data is modified to use atmospheric and oceanographic variables with the same moment as the SIE.
The static metadata is calculated by counting the days from the beginning of time in IceTFT to enhance locality information.
The loss function is modified to use mean square error.

##Scripts are all saved in the main folder, with descriptions below:

* **script\_train\_fixed\_params.py**: Calibrates the TFT using a predefined set of hyperparameters, and evaluates for a given experiment.
* **script\_hyperparameter\_optimisation.py**: Runs full hyperparameter optimization using the default random search ranges defined for the TFT.

##Run the IceTFT

#Step1: Setting expt_settings\configs.py 
#'ice' presents the daily SIE prediction; 'icemonthly' presents the monthly SIE prediction
default_experiments = ['icedaily', 'icemonthly']
Add an entry in ``data_csv_path`` mapping the experiment name to name of the csv file containing the data

#Step2: Downloading data
Download the SIE, atmospheric and oceanographic variables what you need, and modify the '_column_definition' of data_formatters\default_experiments.py
The name of variables  in data_csv file should map the attributes of '_column_definition' 

#Step3: Setting experiment designs
Set the division of the training set, the length of input in data_formatters\default_experiments.py

#Step4: Training and evaluating
For full hyperparameter optimization, run:
python script_hyperparam_opt.py

To train the network with the optimal default parameters, run:
python script_train_fixed_params.py

#Step5: Analyze variable sensitivitie
Add noise to the variables you want to study in the original data, make a new data_csv file, and add noise to only one variable per analysis
To evaluate the change of network with optimal default parameters, run:
python variable_sensitivity.py


