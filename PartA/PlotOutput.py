import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch

#Class names of the dataset
class_names = ["Amphibia","Animalia","Arachnida","Aves","Fungi","Insecta","Mammalia","Mollusca","Plantae","Reptilia"]

def plot_images(images, true_label, predicted_labels):
    #Create subplots of 10 rows representing true labels, 4 columns - 1 for displaying the true lable, other 3 for plotting the input image and displaying the prediction by model
    fig, axs = plt.subplots(10, 4, figsize = (20,30))
    
    #Iterate through the classes
    for i in range(0,10):
        #Get 3 sample input images of class type class_names[i]
        class_images = images[true_label == i][:3]
        #Get the respective predictions
        class_pred_lables = predicted_labels[true_label == i][:3]
        #Display the true label in first column
        axs[i,0].axis('off')
        axs[i,0].text(0.5,0.5, f'True Label: {class_names[i]}', ha = 'center', va = 'center', fontsize = 15)
        
        #Display the respective images and predicted lables in next 3 columns
        for j in range(1,4):
            img = transforms.ToPILImage()(class_images[j-1]).convert("RGB")
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            axs[i, j].set_title(f'Predicted: {class_names[class_pred_lables[j-1]]}',  fontsize = 15)


    plt.tight_layout()
    #Save the plot as a png file
    plt.savefig("TestPredictionPlot.png")
    #plt.show()


    return
