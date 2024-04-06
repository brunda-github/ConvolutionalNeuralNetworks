import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
from cnn import CNNModel
from DataModule import CustomDataModule
import pytorch_lightning as pl
from PIL import Image
import PlotOutput
import os
import random
import wandb
import matplotlib.pyplot as plt
import torch
import argparse


#Define tranforms with data augmentation
dataaugmentation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.3868, 0.4566,0.4651], std = [0.2283, 0.2173, 0.2269]), #mean of the input data calculated using module calculate_mean
    transforms.Resize((256, 256)),
    transforms.RandomRotation(degrees=10),  # Rotate the image randomly up to 10 degrees
    transforms.RandomHorizontalFlip(),      # Flip the image horizontally with a 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
])

#Transform without data augmentation
transform = transforms.Compose([transforms.ToTensor(),  #Covert to Tensor
                                transforms.Normalize(mean = [0.3868, 0.4566,0.4651], std = [0.2283, 0.2173, 0.2269]), #mean of the input data calculated using module calculate_mean
                                transforms.Resize((256, 256))])

#Transform for test data without normalisation
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

#Path for train data - To be updated as required
trainDataDir = "D:\\brunda\\inaturalist_12K\\train"
trainData = ImageFolder(root = trainDataDir, transform=transform)

#Path for test data - To be updated as required
testDataDir = "D:\\brunda\\inaturalist_12K\\val"
testData = ImageFolder(root = testDataDir, transform=transform)



#Split the training and val data with equal count of each class

#Step 1: Get a list of indices for each class in the dataset
class_indices = {}
for idx, (_,label) in enumerate(trainData):
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)


#Step 2: Calculate size of train and val data per class and the data will be loaded in train_model module using DataModule of pytorch lightning 
train_indices = []
val_indices = []
for lable, indices in class_indices.items():
    random.shuffle(indices)
    size = len(indices)
    train_size = int(0.8*size)
    train_indices.extend(indices[:train_size])
    val_indices.extend(indices[train_size:])



def calculate_mean():
    # Calculate mean of the input data. Done during initial setup and used the outputs obtained later to reduce redundant computations
    mean = torch.mean(trainData.data, axis=(0, 1, 2)) / 255.0  # Normalize mean to [0,1]
    std = torch.std(trainData.data, axis=(0, 1, 2)) / 255.0  # Normalize std to [0,1]

def train_model(config =None, args = None):
    
    wandb.init(config = config, project=args.wandb_project, entity=args.wandb_entity)
    config = wandb.config
    
    #run_name = "Sample_Prediction_plot"
    run_name = "CNN_output_fsize_{}_nfilters_{}_fStrategy_{}_activationfunc_{}_bn_{}_dropout_{}_lr_{}_epochs_{}_batchsize_{}".format(config.convfiltersize, config.nFilters, config.filtersStrategy, config.activation_func, config.BatchNormalization, config.dropout_prob, config.learningRate, config.epochs, config.batch_size)
    wandb.run.name = run_name
    
    #Split the training and val data with equal count of each class using the below steps

    #Step 1: Get a list of indices for each class in the dataset
    class_indices = {}
    for idx, (_,label) in enumerate(trainData):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)


    #Step 2: Calculate size of train and val data per class and the data will be loaded in train_model module using DataModule of pytorch lightning 
    train_indices = []
    val_indices = []
    for lable, indices in class_indices.items():
        random.shuffle(indices)
        size = len(indices)
        train_size = int(0.8*size)
        train_indices.extend(indices[:train_size])
        val_indices.extend(indices[train_size:])
    
    #Step 3: Create Subset
    train_data = Subset(trainData, train_indices)
    val_data = Subset(trainData, val_indices)

    #Step 4: Create DataLoaders
    test_loader = DataLoader(testData, batch_size = config.batch_size, shuffle = False)
    # Create LightningDataModule instance
    data_module = CustomDataModule(train_data, val_data, batch_size=config.batch_size)

    # Initialize Lightning Trainer with wandb logger.
    trainer = pl.Trainer(max_epochs=config.epochs, logger=pl.loggers.WandbLogger())

    # Create an instance of the SimpleCNN model
    model = CNNModel(convfiltersize = config.convfiltersize, nFilters = config.nFilters, filtersStrategy = config.filtersStrategy, activationfunction = config.activation_func, enableBatchNormalisation = config.BatchNormalization, dropout_prob = config.dropout_prob, learningRate = config.learningRate)

    # Train the model using Lightning Trainer
    trainer.fit(model, data_module)

    #Test the model with test data
    test_results = trainer.test(model, test_loader)
    #print(test_results)
    
    #Plot sample predictions on test data
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    test_Data = ImageFolder(root = testDataDir, transform=test_transform)
    dataloader = torch.utils.data.DataLoader(test_Data, batch_size = 1, shuffle = False)
    #Create a list of images and true_labels from test data
    images = []
    true_labels = []
    for image, label in dataloader:
        images.append(image)
        true_labels.append(label)

    #Convert the labels to tensors
    images = torch.cat(images, dim =0)
    true_labels = torch.tensor(true_labels)
    pred_labels = torch.tensor(model.pred)
    
    #Invoke the plot_images module
    PlotOutput.plot_images(images, true_labels, pred_labels)
    #Load the saved image and log it in wandb
    img2 = plt.imread("TestPredictionPlot.png")
    wandb.log({"TestPredictionPlot": wandb.Image(img2)})
    
    #Finish the run
    wandb.finish()

def sweeps():
    #module to run sweeps
    #Define sweep config
    sweep_config = { "name" : "CNN","method": "bayes"}
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

if __name__ == "__main__":
    #Update the login key if the project visibility is not open
    #wandb.login(key = "")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train neural network with specified configurations')
    parser.add_argument('--wandb_project', '-wp', type=str, default='', help='Weights & Biases project name') #basic-intro
    parser.add_argument('--wandb_entity', '-we', type=str, default='', help='Weights & Biases entity') #drbruap
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training')
    parser.add_argument('--BatchNormalization', '-bn', type=str, default=True, choices=[True, False], help='Batch Normalization')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_prob', '-dp', type=float, default=0.2, help='Droput probability')
    parser.add_argument('--nFilters', '-nF', type=int, default=64, help='Number of filters')
    parser.add_argument('--convfiltersize', '-cf', type=int, default=3, help='Size of convolutional layers')
    parser.add_argument('--activation', '-a', type=str, default='tanh', choices=['Mish', 'PReLU', 'LeakyReLU' ,'ReLU'], help='Activation function')
    parser.add_argument('--filtersStrategy', '-fs', type=str, default='Double', choices=['Same', 'Halve', 'Double'], help='Strategy for calculating num of filters in next convolutional layer')

    args = parser.parse_args()

    config = {
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "convfiltersize" :args.convfiltersize,
    "nFilters" : args.nFilters,
    "learningRate" : args.learning_rate,
    "filtersStrategy" : args.filtersStrategy,
    "activation_func" : args.activation,
    "BatchNormalization" : True,
    "dropout_prob" : args.dropout_prob
    }
    train_model(config, args)