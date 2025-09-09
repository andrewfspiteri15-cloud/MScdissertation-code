from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import torchvision
from torchvision import models, transforms
#import matplotlib.pyplot as plt
#import time
import os
#import copy
#from sklearn.model_selection import KFold
import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
#import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split
#import seaborn as sn

#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
#from torchvision.io import read_image
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DanceDataset(Dataset): #custom dataset object https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file, header=None)
        self.img_labels = self.data.iloc[:,2]
        self.img_dir = self.data.iloc[:,3]
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path =self.img_dir.iloc[idx]
        image = Image.open(img_path)
        #image = read_image(img_path) / 255 #dividing by 255 here if not using ToTensor(normalizing)
        #for easier model convergence (less time taken doing dot products)
        label = self.img_labels.iloc[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
 #data_dir_train="./dancers1and2", data_dir_test="./dancers3"


##Loading the datasets (UNUSED)
#def load_data(file1, file2):
#    transform = transforms.Compose([
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#    ])
    
#    from dancedataset import DanceDataset
    #trainset = datasets.ImageFolder(
    #    root=data_dir_train, transform=transform)
#    trainset = DanceDataset(file1)

    #testset = datasets.ImageFolder(
    #    root=data_dir_test, transform=transform)
#    testset = DanceDataset(file2)
#    
#    return trainset, testset

##FINDING MEAN AND STANDARD DEVIATION OF EACH CHANNEL THAT WE WILL USE TO NORMALIZE THE IMAGES
#WE WILL CALCULATE THIS OFF OF THE TRAINING SET AND USE THEM FOR THE TRAINING/VALIDATION AND TEST SETS
file1 = 'C:\\Users\\Andrew\\Downloads\\Masters Data\\PostureRecognition - The third folder\\data_old_labels_dancers_1and2.csv'

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor()
])

trainset = DanceDataset(file1, transform = transform)
trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)

data = next(iter(trainloader))

#The mean has to be calculated over all the images, their height, and their width, however, not over the channels.
#In the case of colored images, an output tensor of size 3 is expected. Reference https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

get_mean_and_std(trainloader)
#returns (tensor([0.4742, 0.4794, 0.4527]), tensor([0.2028, 0.2015, 0.2052])) for dancers1and3 RESIZED TO 224X224 FOR RESNET


#specifically for Alexnet to change fully connected layer at end to classify for 7 labels
#Net.classifier._modules['6'] = nn.Linear(4096, 7)

def train_model(config, checkpoint_dir=None):
    # manually rewrite which CNN architecture to use (make sure to change output layer neurons)
    net = models.resnet34(pretrained=False)
    net.fc = nn.Linear(512,7) # since we have 7 labels
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net = net.to(device)

    #criterion = nn.CrossEntropyLoss()
    
    ## Manually set to which optimization algorithm to run
    optimizer = optim.AdamW(net.parameters(), lr=config["lr"], betas = (config["beta1"],config["beta2"]), weight_decay = config["l2"])
    #optimizer = optim.AdamW(net.parameters(), lr=config["lr"], weight_decay = config["l2"])
    #optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay = config["l2"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    #trainset, testset = load_data("data_old_labels_dancers_1and2.csv","data_old_labels_dancer_3.csv")
    #trainset, testset = load_data("Downloads\Masters Data\PostureRecognition - The third folder\data_old_labels_dancers_1and2.csv","Downloads\Masters Data\PostureRecognition - The third folder\data_old_labels_dancer_3.csv")
    #^ remember that this will give an error in the dataloader since the image paths of the images assume 
    
    ## Manually set if I am resizing or not. The normalization values are edited here using def get_mean_and_std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4771, 0.4808, 0.4549], [0.1979, 0.1966, 0.2003])
    ])
    
    from dancedataset import DanceDataset
    #trainset = datasets.ImageFolder(
    #    root=data_dir_train, transform=transform)
    
    #file1 = os.path.abspath("./data_old_labels_dancers_1and2.csv") #get absolute path
    #file2 = os.path.abspath("./data_old_labels_dancer_3.csv")
    
    file1 = 'C:\\Users\\Andrew\\Downloads\\Masters Data\\PostureRecognition - The third folder\\data_old_labels_dancers_1and2.csv'
    #file2 = 'C:\\Users\\Andrew\\Downloads\\Masters Data\\PostureRecognition - The third folder\\data_old_labels_dancer_3.csv'
    
    trainset = DanceDataset(file1, transform = transform)

    #testset = datasets.ImageFolder(
    #    root=data_dir_test, transform=transform)
    #testset = DanceDataset(file2, transform = transform)
    
    ## for weighted cross entropy https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss https://androidkt.com/how-to-use-class-weight-in-crossentropyloss-for-an-imbalanced-dataset/ https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/8
    y1 = trainset.img_labels
    class_weights=class_weight.compute_class_weight('balanced',np.unique(y1),y1)
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    criterion = nn.CrossEntropyLoss(weight = class_weights)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2)
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2)

    for epoch in range(50):  # loop over the dataset multiple times
        
        running_loss = 0.0
        epoch_steps = 0
        
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 10 == 9:  # print every 10 mini-batches
                print("[%d, %5d] train loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                print(running_loss, epoch_steps)
                running_loss = 0.0
        
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        
        net.eval()
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")



def test_accuracy(best_trial):
    # manually rewrite which CNN architecture to use (make sure to change output layer neurons)
    net = models.resnet34(pretrained=False)
    net.fc = nn.Linear(512,7)
    
    #criterion = nn.CrossEntropyLoss()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net = net.to(device)
    
    #trainset, testset = load_data("data_old_labels_dancers_1and2.csv","data_old_labels_dancer_3.csv")
    #trainset, testset = load_data("Downloads\Masters Data\PostureRecognition - The third folder\data_old_labels_dancers_1and2.csv","Downloads\Masters Data\PostureRecognition - The third folder\data_old_labels_dancer_3.csv")
    from dancedataset import DanceDataset
    #trainset = datasets.ImageFolder(
    #    root=data_dir_train, transform=transform)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4771, 0.4808, 0.4549], [0.1979, 0.1966, 0.2003])
    ])
    
    #file1 = os.path.abspath("./data_old_labels_dancers_1and2.csv") #get absolute path
    #file2 = os.path.abspath("./data_old_labels_dancer_3.csv")
    
    file1 = 'C:\\Users\\Andrew\\Downloads\\Masters Data\\PostureRecognition - The third folder\\data_old_labels_dancers_1and2.csv'
    file2 = 'C:\\Users\\Andrew\\Downloads\\Masters Data\\PostureRecognition - The third folder\\data_old_labels_dancer_3.csv'
    
    trainset = DanceDataset(file1, transform = transform)

    #testset = datasets.ImageFolder(
    #    root=data_dir_test, transform=transform)
    testset = DanceDataset(file2, transform = transform)
    
    ## for weighted cross entropy https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss https://androidkt.com/how-to-use-class-weight-in-crossentropyloss-for-an-imbalanced-dataset/ https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/8
    y1 = trainset.img_labels
    class_weights=class_weight.compute_class_weight('balanced',np.unique(y1),y1)
    class_weights=torch.tensor(class_weights,dtype=torch.float)

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    net.load_state_dict(model_state)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    test_running_loss = 0.0
    
    pred_test_labels = []
    true_test_labels = []
    
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            
            pred_test_labels.extend(predicted)
            true_test_labels.extend(labels)
    
    
    #added confisuion matrix
    #cf_matrix_test = confusion_matrix(labels, predicted)
    
    print('Test accuracy: %d %%' % (100.0 * (correct / total)))
    print('Test loss: %.3f' % (test_running_loss / len(testloader.sampler)))
    
    pred_test_labels = torch.stack(pred_test_labels)
    true_test_labels = torch.stack(true_test_labels)
        
    cf_matrix_test = confusion_matrix(true_test_labels, pred_test_labels)
    
    print(cf_matrix_test)
    
    return correct / total


def main(num_samples=20, max_num_epochs=50, gpus_per_trial=2):
    #data_dir = os.path.abspath("./data")
    config = {
        "l2": tune.loguniform(1e-6, 1e-1),
        "lr": tune.loguniform(1e-6, 1),
        "batch_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        #"momentum": tune.uniform(0.01, 0.999) # if using SGD
        "beta1": tune.uniform(0.01, 0.999), # if using AdamW
        "beta2": tune.uniform(0.01, 0.999) # if using AdamW
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=2,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2" "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        tune.with_parameters(train_model),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    test_acc = test_accuracy(best_trial)
        
    print("Best trial test set accuracy: {}".format(test_acc))


## Number of samples to take and epochs to run them over
main(num_samples=20, max_num_epochs=50, gpus_per_trial=0)

