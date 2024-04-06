import pytorch_lightning as pl
import torch

# Create LightningDataModule for data handling
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32):
        super().__init__()
        #Initialize the train and val data set to load the data later
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        #This method is invoked by lightning module at the beginning of the training.
        #Shuffle parameter is set to true, inorder to generalize the model by not remembering the sequence of images
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        #This method is invoked by lightning module at the beginning for the first epoch.
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
