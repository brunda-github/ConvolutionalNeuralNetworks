import argparse
from CNN_Pretrained import PretrainedModel
from CustomDataModule import CustomDataModule
import pytorch_lightning as pl
import wandb

if __name__ == "__main__":
    #Update the login key if the project visibility is not open
    #wandb.login(key = "")
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train neural network with specified configurations')
    parser.add_argument('--wandb_project', '-wp', type=str, default='basic-intro', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', '-we', type=str, default='drbruap', help='Weights & Biases entity')
    args = parser.parse_args()
    
    wandb.init(project = args.wandb_project, entity = args.wandb_entity , id = "Pretrained_InceptionV3")
    
    model = PretrainedModel(learningRate = 0.001, batch_size = 32, epochs = 5, preTrainedModel = "InceptionV3")
    data = CustomDataModule(preTrainedModel = "InceptionV3")
    trainer = pl.Trainer(max_epochs = 5, logger=pl.loggers.WandbLogger(), accelerator = "gpu")

    trainer.fit(model, datamodule = data)
    test_results = trainer.test(datamodule = data)

    print(test_results)
