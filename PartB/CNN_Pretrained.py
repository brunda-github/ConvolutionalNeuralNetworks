import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from statistics import mean

class PretrainedModel(pl.LightningModule):
    def __init__(self, learningRate = 0.001, batch_size = 32, epochs = 10, preTrainedModel = "InceptionV3"):
        #Invoke lightning module init method
        super().__init__()
        
        #Initialize input params
        self.learning_rate = learningRate
        self.batch_size = batch_size
        self.epochs = epochs
        self.preTrainedModel = preTrainedModel

        #Load pretrained models
        self.model = torchvision.models.inception_v3(pretrained = True).to(self.device)
        if preTrainedModel == "resnet50":
          self.model = torchvision.models.resnet50(pretrained = True)
          
        #Replace the fully connected/output layer with the no.of output classes as required
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        #Adding softmax layer
        self.softmax = nn.Softmax(dim = 1)

        #Freeze convolution layers
        for param in self.model.parameters():
            param.requires_grad = False

        #Unfreeze the fully connected layers for fine tuning
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        #Log the hyper params
        self.save_hyperparameters()
        
        return

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def training_step(self,batch, idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        output = outputs
        #Inception models return InceptionOutputs struct as output,
        #while other models liek resnet outputs tensor
        #So getting the tensor output if the output is not a tensor (like inception models)
        if not torch.is_tensor(outputs):
            output = outputs[0]

        #Backprogation is done automatically after loss is calculated
        loss = self.criterion(output, labels)
        self.train_loss.append(loss.item())

        #Calculate accuracy
        _, pred = torch.max(output, 1)
        accuracy = (pred == labels).sum().item() / len(labels)
        
        self.train_acc.append(accuracy)
        return loss

    def on_train_epoch_end(self):
        #Since the data is trained in batches, on the end of train epoch, 
        #calculate the average of metrics calculated for every batch
        super().on_train_epoch_end()
        total_acc = mean(self.train_acc)
        total_loss = mean(self.train_loss)
       
        print("Train_Accuracy", total_acc)
        print("Train_Loss", total_loss)

        #self.log logs the metrics to wandb of wandblogger is used in lightning module trainer instance
        metrics = {'epoch':self.current_epoch, 'Train_Loss':total_loss, 'Train_Accuracy':total_acc}
        self.log_dict(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.log('Train_Accuracy', total_acc)
            self.log('Train_Loss', total_loss)

        self.train_acc = []
        self.train_loss = []
        return

    def validation_step(self, batch, idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        output = outputs
        #Inception models return InceptionOutputs struct as output,
        #while other models liek resnet outputs tensor
        #So getting the tensor output if the output is not a tensor (like inception models)
        if not torch.is_tensor(outputs):
            output = outputs[0]

        #Lightning module makes sure that bBackprogation is not done during validation step
        loss = self.criterion(output, labels)
        self.val_loss.append(loss.item())

        #Calculate accuracy
        _, pred = torch.max(output, 1)
        accuracy = (pred == labels).sum().item() / len(labels)
        self.val_acc.append(accuracy)
        return loss

    def on_validation_epoch_end(self):
        #Since the data is validated in batches, on the end of every epoch, 
        #calculate the average of metrics calculated for every batch
        super().on_validation_epoch_end()
        total_acc = mean(self.val_acc)
        total_loss = mean(self.val_loss)
        
        print("Val_Accuracy", total_acc)
        print("Val_Loss", total_loss)

        #self.log logs the metrics to wandb of wandblogger is used in lightning module trainer instance
        metrics = {'epoch':self.current_epoch, 'Val_Loss':total_loss, 'Val_Accuracy':total_acc}
        self.log_dict(metrics)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.log('Val_Accuracy', total_acc)
            self.log('Val_Loss', total_loss)

        self.val_acc = []
        self.val_loss = []
        return

    def test_step(self, batch, idx):
        inputs, labels = batch
        outputs = self.model(inputs)

        output = outputs
        if not torch.is_tensor(outputs):
            output = outputs[0]

        test_loss = self.criterion(output, labels)
        self.log('Test_Loss', test_loss)

        #Calculate accuracy
        _, pred = torch.max(output, 1)
        accuracy = (pred == labels).sum().item() / len(labels)
        self.log('Test_Accuracy', accuracy)

        return accuracy

    def configure_optimizers(self):
        #this method is invoked to initilaze the optimizer during the initial setup by lightning module's trainer instance
        optimizer = optim.Adam(self.model.fc.parameters(), lr = self.learning_rate)
        return optimizer