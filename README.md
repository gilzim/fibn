# FiBN: Visual Reasoning with a General Conditioning Layer and Conditional Batch Normalization

### Src Folder Structure

This code is a fork from the code for "FiLM: Visual Reasoning with a General Conditioning Layer" available [here](https://github.com/ethanjperez/film).

#### environment & requirements
Both environment.yml and requirements.txt are used during the setup to prepare the virtual environment for the code

#### vr
The vr folder includes .py files for preprocessing data, a utils.py file for loading models from checkpoints
and the models package which contatins the implementations of the different layers and models.
##### Important files
- Our CBN layer is located in vr/models/cbn_layer.py
- The linguistic pipeline(GRU) is located in vr/models/film_gen.py
- The visual pipeline(FiBN) is located in vr/models/filmed_net.py

#### scripts
The scripts folder includes .py files for preprocessing the data and for training and running models 
and the train folder which contains .sh scripts for training the FiBN model.
##### Important files
- The train.py implaments the training loop
- The run_model.py is used to run the model from a given checkpoint on samples

#### img
The img folder includes an example picture of the CLEVR dataset and the stats folder which contains gammas and betas distributions of the FiBN model.

### Important Notes
- The code can only run on the Linux OS
- All bash scripts and commands must be executed from the src folder

### Setup
First, create a conda enviorment using the provided environment.yml file and run the following command: 
```bash
pip install -r requirements.txt
```

Second, run the download_dataset.sh script located in scripts folder to download the CLEVR dataset.

Third, run the preprocess_data.sh script located in scripts folder to preprocess the data.

### Training
The below script has the hyperparameters and settings to reproduce FiBN CLEVR results:
```bash
sh scripts/train/fibn.sh
```
The above script must use **2** Gpus.

Training a FiBN CLEVR model should take ~20 hours on 2 average GPUs.

### Running models
Any script/command must use only **1** GPU

There is an interactive command line tool for use with the below command/script.
```bash
python run_model.py --program_generator <model checkpoint> --execution_engine <model checkpoint>
```
When both checkpoint paths are the same.

By default, the command runs on [this CLEVR image](https://github.com/gilzim/fibn/blob/master/img/CLEVR_val_000017.png), but you may modify which image to use via command line flag to test on any CLEVR image.

We added the script run_model_fibn.sh which runs on a batch of 3000 samples. It returns the accuracy of the model
and saves different gammas & bettas distributions in img/stats.
