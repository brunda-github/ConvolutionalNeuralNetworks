import torch
#from torchmetrics.functional import Accurcay, Loss
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from statistics import mean
import wandb

class CNNModel(pl.LightningModule):
    def __init__(self, inputsize = 256, nConvLayers = 5, convfiltersize = 3, nFilters = 32, filtersStrategy = "Same", activationfunction = "ReLU", enableBatchNormalisation = True, dropout_prob = 0.0, learningRate = 1e-3):
        super(CNNModel, self).__init__()

        self.convfiltersize = convfiltersize
        self.activationfunction = activationfunction
        self.lr = learningRate
        poolingfiltersize = 2

        #Define model architecture
        layers = []

        in_channels = 3  # Initial input channels (RGB)
        out_channels = []

        for i in range(0, nConvLayers):
            out_channels.append(convfiltersize)
            if(filtersStrategy == "Halve") and convfiltersize >= 2:
                convfiltersize = convfiltersize // 2
            elif filtersStrategy == "Double" and convfiltersize < 64 :
                convfiltersize *= 2
            #Do nothing if strategy is to use same number of filters

        self.inputsize = inputsize
        input_size = inputsize
        self.finalconvLayeroutputsize = inputsize

        for idx, channels in enumerate(out_channels):
            # 1. Convolution layer
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=self.convfiltersize, padding=1))
            convOutputsize = input_size -  self.convfiltersize + 2*1 + 1 #Assuming stride = 1

            if enableBatchNormalisation:
                layers.append(nn.BatchNorm2d(channels))

            #2.Activation layer
            if activationfunction == "Mish":
                layers.append(nn.Mish())
            elif activationfunction == "PReLU":
                layers.append(nn.PReLU())
            elif activationfunction == "LeakyReLU":
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())

            #3. Max-pooling layer
            poolingInputSize = convOutputsize
            layers.append(nn.MaxPool2d(kernel_size=poolingfiltersize, stride=2))
            poolingOutputsize = ((poolingInputSize - poolingfiltersize) // 2) + 1

            self.finalconvLayeroutputsize = poolingOutputsize * poolingOutputsize * channels # since all the inputs and filters are considered square shaped, h = w

            # Update input channels for the next layer
            in_channels = channels
            # Update input_size for next conv layer
            input_size = poolingOutputsize

        #Flatten before fully connected layer
        layers.append(nn.Flatten())

        # Define fully connected layers
        layers.append(nn.Linear(self.finalconvLayeroutputsize, 128))
        layers.append(nn.ReLU())
        if dropout_prob != 0.0:
                layers.append(nn.Dropout(p = dropout_prob))
        layers.append(nn.Linear(128, 10))  # Output layer with 10 classes (change as needed)
        #layers.append(nn.Softmax(dim = 1))

        # Convert the list of layers into a Sequential container
        self.model = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim = 1)

        #if enableLearningratescheduler:
        #    self.scheduler = StepLR()

        # Metrics for epochs
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

        #Final metrics
        self.final_train_acc = []
        self.final_val_acc = []
        self.final_train_loss = []
        self.final_val_loss = []

        self.to('cuda')

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)

    def training_step(self, batch, batch_idx):

        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)

        #Calculate accuracy
        _, pred = torch.max(outputs, 1)
        accuracy = (pred == labels).sum().item() / len(labels)

        #self.log('train_accuracy', accuracy)
        #self.log('train_loss', loss.item())

        self.train_acc.append(accuracy)
        self.train_loss.append(loss.item())

        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train_acc = []
        self.train_loss = []
        return

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        total_acc = mean(self.train_acc)
        total_loss = mean(self.train_loss)
        #total_loss = torch.mean(torch.stack(self.train_loss))
        print("Train_Accuracy", total_acc)
        print("Train_Loss", total_loss)

        metrics = {'epoch':self.current_epoch, 'Train_Loss':total_loss, 'Train_Accuracy':total_acc}
        self.log_dict(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.log('Train_Accuracy', total_acc)
            self.log('Train_Loss', total_loss)

        self.final_train_acc.append(total_acc)
        self.final_train_loss.append(total_loss)
        return

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)

        #Calculate accuracy
        _, pred = torch.max(outputs, 1)
        accuracy = (pred == labels).sum().item() / len(labels)

        self.val_acc.append(accuracy)
        self.val_loss.append(loss.item())

        return loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_acc = []
        self.val_loss = []
        return

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        total_acc = mean(self.val_acc)
        total_loss = mean(self.val_loss)
        #total_loss = torch.mean(torch.stack(self.val_loss))
        self.final_val_acc.append(total_acc)
        self.final_val_loss.append(total_loss)
        print("val_Accuracy", total_acc)
        print("val_Loss", total_loss)
        metrics = {'epoch':self.current_epoch, 'Val_Loss':total_loss, 'Val_Accuracy':total_acc}
        self.log_dict(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.log('Val_Accuracy', total_acc)
            self.log('Val_Loss', total_loss)
        return

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)

        #Calculate accuracy
        _, pred = torch.max(outputs, 1)
        accuracy = (pred == labels).sum().item() / len(labels)
        self.pred = pred
        #print("Accuracy", accuracy)
        self.log('Test_Accuracy', accuracy)
        self.log('Test_Loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        #return torch.optim.SGD(self.parameters(), lr = 0.01)
