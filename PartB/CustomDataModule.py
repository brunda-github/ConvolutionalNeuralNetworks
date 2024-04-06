import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import pytorch_lightning as pl

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size = 32, preTrainedModel = "InceptionV3"):
        super().__init__()
        self.batch_size = batch_size
        self.preTrainedModel = preTrainedModel

    def setup(self, stage = None):
        #Resize input images according to the model
        if self.preTrainedModel == "InceptionResNetV2" or self.preTrainedModel == "InceptionV3":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), transforms.Resize((299, 299))])#Using imagenet statistics for mean and std
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), transforms.Resize((224, 224))])#Using imagenet statistics for mean and std
            
        #Update the train and test paths as required
        trainDataDir = "D:\\brunda\\inaturalist_12K\\train"
        trainData = ImageFolder(root = trainDataDir, transform=transform)
        
        testDataDir = "D:\\brunda\\inaturalist_12K\\val"
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
            size = len(indices)
            train_size = int(0.8*size)
            train_indices.extend(indices[:train_size])
            val_indices.extend(indices[train_size:])

        #Step 3: Create Subset
        train_data = Subset(trainData, train_indices)
        val_data = Subset(trainData, val_indices)

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = testData

        return

    def train_dataloader(self):
        #dataloader to load train data - Invoked by lightning module automatically when training starts
        #shuffle parameter is set to true to generalize model and not get biased on training input sequence
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        #dataloader to load validation data - Invoked by lightning module automatically
        return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.batch_size, shuffle = False)

    def test_dataloader(self):
        #dataloader to load test data - Invoked by lightning module automatically
        return torch.utils.data.DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)

