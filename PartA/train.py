import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from CNN import CNNModel
from DataModule import CustomDataModule
import pytorch_lightning as pl
from PIL import Image
import os
import random
import wandb

wandb.login(key = "017101a520090630fd58ad8684de73bf54c45117")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.3868, 0.4566,0.4651], std = [0.2283, 0.2173, 0.2269]), transforms.Resize((256, 256))])#mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]) #Using imagenet statistics
trainDataDir = "C:\\Brunda\\Learning\\Python\\DL\\nature_12K\\inaturalist_12K\\train"
trainData = ImageFolder(root = trainDataDir, transform=transform)
#trainData = ImageFolder(root = trainDataDir, transform = )
testDataDir = "C:\\Brunda\\Learning\\Python\\DL\\nature_12K\\inaturalist_12K\\val"
testData = ImageFolder(root = testDataDir, transform=transform)

#Split the training and val data with equal count of each class

#Step 1: Get a list of indices for each class in the dataset
class_indices = {}
for idx, (_,label) in enumerate(trainData):
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)


#Step 2: Calculate size of train and val data per class

train_indices = []
val_indices = []
for lable, indices in class_indices.items():
    random.shuffle(indices)
    size = len(indices)
    train_size = int(0.9*size)
    train_indices.extend(indices[:train_size])
    val_indices.extend(indices[train_size:])


def train_model(config =None):
    wandb.init(config = config)
    config = wandb.config
    run_name = "CNN_fsize_{}_nfilters_{}_fStrategy_{}_activationfunc_{}_bn_{}_dropout_{}_lr_{}_epochs_{}_batchsize_{}".format(config.convfiltersize, config.nFilters, config.filtersStrategy, config.activation_func, config.BatchNormalization, config.dropout_prob, config.learningRate, config.epochs, config.batch_size)
    wandb.run.name = run_name
    #Step 3: Create Subset
    train_data = Subset(trainData, train_indices)
    val_data = Subset(trainData, val_indices)

    #Step 4: Create DataLoaders
    test_loader = DataLoader(testData, batch_size = config.batch_size, shuffle = False)
    # Create LightningDataModule instance
    data_module = CustomDataModule(train_data, val_data, batch_size=config.batch_size)

    # Initialize Lightning Trainer
    trainer = pl.Trainer(max_epochs=config.epochs, logger=pl.loggers.WandbLogger())

    # Create an instance of the SimpleCNN model
    model = CNNModel(convfiltersize = config.convfiltersize, nFilters = config.nFilters, filtersStrategy = config.filtersStrategy, activationfunction = config.activation_func, enableBatchNormalisation = config.BatchNormalization, dropout_prob = config.dropout_prob, learningRate = config.learningRate)

    # Train the model using Lightning Trainer
    trainer.fit(model, data_module)

    test_results = trainer.test(model, test_loader)
    print(model.pred)

    print(test_results)

if __name__ == "__main__":
    sweep_config = { "name" : "CNN","method": "random"}
    metric = {
    "name" : "Val_Accuracy",
    "goal" : "maximize"
    }
    sweep_config["metric"] = metric
    parameters_dict = {
    "epochs" : {"values" : [5,10]},
    "batch_size" : {"values":[32,64]},
    "convfiltersize" :{"values" : [3,5]},
    "nFilters" : {"values":[32,64]},
    "learningRate" : {"values":[1e-3]},
    "filtersStrategy" : {"values":["Same", "Halve", "Double"]},
    "activation_func" : {"values" : ["LeakyReLU", "PReLU", "Mish", "ReLU"]},
    "BatchNormalization" : {"values":[True]},
    "dropout_prob" : {"values" : [0.0, 0.2, 0.3]}
    }
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="CNN")
    wandb.agent(sweep_id, train_model, count=20)

