# Fine tuning a pretrained model

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
