# ConvolutionalNeuralNetworks

This repo contains 2 folders 
1. Part A - Training a CNN from scratch using iNaturalist dataset
2. Part B - Using pretrained models, freezing convolutional layers and fine tuning the fully connected layers

Below are the instructions to run/train the respective models

# Training CNN from scratch

## Requirements
The following packages are required to train this model
1. torch
2. torchvision
3. pytorch_lightning
4. wandb
5. matplotlib

**Note**: Install cuda supported torch and torch vision packages or use google colab which by default provides cuda support packages


## Steps to train the model
1. Install the required python packages as metioned above. **Note**: Make sure the run environment has GPU with cuda toolkit
2. Download the python modules (cnn.py, DataModule.py, PlotOutput.py, train_parta.py)
3. Update the train and test data dir variables (trainDataDir, testDataDir) to appropriate dataset paths in train_parta.py
4. Run the command by replacing myname myprojectname respectively
#### python train_parta.py --wandb_entity myname --wandb_project myprojectname
Note: Make sure wandb project visibility is open. If not, make sure to call wandb.login() before initialising wandb runs

5. Following command line arguments are supported for train_parta.py
This python file can be executed to train a FFN model by passing required arguments as mentioned below

| Name                | Description                                                                           |
|---------------------|---------------------------------------------------------------------------------------|
| `wp`, `wandb_project` | Project name used to track experiments in Weights & Biases dashboard                |
| `we`, `wandb_entity`  | Wandb Entity used to track experiments in the Weights & Biases dashboard             |
| `e`, `epochs`          | Number of epochs to train the neural network                                         |
| `b`, `batch_size`      | Batch size used for training the neural network                                      |
| `bn`, `BatchNormalization` | Enable or disable batch normalization                                            |
| `lr`, `learning_rate`  | Learning rate used for optimizing model parameters                                    |
| `dp`, `dropout_prob`   | Dropout probability in the fully connected layer                                        |
|`nF`, `nFilters`        |Number of filters                                                                      |
|`cf`,`convfiltersize`   | Size of convolutional layers filter                                                    |
| `a`, `activation`      | Activation function used (`ReLU`, `Mish`, `PReLU`, `LeakyReLU`)                       |
|`fs`,`filtersStrategy`| Strategy for calculating num of filters in next convolutional layer|


---------------------------------------------------------

1. cnn.py - This file contains class CNNModel built using pytorch_lightning module and provies the flexibility in defining the architechture to train the model
2. DataModule.py - This file defines CustomDataModule class which will be used be used by pytorch_lightning trainer instance to train and test the model.
3. PlotOutput.py - This module is used to plot a sample 10 x 3 grid of 3 sample output predictions for each class type
4. train_parta.py - Used to execute the implementation
   
--------------------------------------------------------------------------

# Fine tuning a pretrained model

pretrained models resnet50 and InceptionV3 can be finetuned in the given implementations.

## Requirements
1. torch
2. torchvision
3. pytorch_lightning
4. wandb
   
**Note**: Install cuda supported versions of torch and torchvision. If not use google colab which by default provides the cuda supported packages.

## Steps to train the model
1. Install the required python packages as metioned above. **Note**: Make sure the run environment has GPU with cuda toolkit
2. Download the python modules (CNN_Pretrained.py, CustomDataModule.py, train_partb.py)
3. Update the train and test data dir variables (trainDataDir, testDataDir) to appropriate dataset paths in setup module in CustomDataModule.py
5. Run the command by replacing myname myprojectname respectively
#### python train_partb.py --wandb_entity myname --wandb_project myprojectname
Note: Make sure wandb project visibility is open. If not, make sure to call wandb.login() before initialising wandb runs

---------------------------------------
Contents:
1. CustomDataModule.py - Used to load the data and return data loaders for lightning module to train
2. CNN_Pretrained.py - Contains class to load a pretrained model InceptionV3 by default/resnet50 if wanted during initialization
3. train_partb.py - creates a customdatamodule and uses it to finetune a pretrained model.

# Fine tuning a pretrained model

pretrained models resnet50 and InceptionV3 can be finetuned in the given implementations.

## Requirements
1. torch
2. torchvision
3. pytorch_lightning
4. wandb
   
**Note**: Install cuda supported versions of torch and torchvision. If not use google colab which by default provides the cuda supported packages.

## Steps to train the model
1. Install the required python packages as metioned above. **Note**: Make sure the run environment has GPU with cuda toolkit
2. Download the python modules (CNN_Pretrained.py, CustomDataModule.py, train_partb.py)
3. Update the train and test data dir variables (trainDataDir, testDataDir) to appropriate dataset paths in setup module in CustomDataModule.py
5. Run the command by replacing myname myprojectname respectively
#### python train_partb.py --wandb_entity myname --wandb_project myprojectname
Note: Make sure wandb project visibility is open. If not, make sure to call wandb.login() before initialising wandb runs

---------------------------------------
Contents:
1. CustomDataModule.py - Used to load the data and return data loaders for lightning module to train
2. CNN_Pretrained.py - Contains class to load a pretrained model InceptionV3 by default/resnet50 if wanted during initialization
3. train_partb.py - creates a customdatamodule and uses it to finetune a pretrained model.
