#import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.utils import class_weight
#import seaborn as sn
from torch.utils.data import random_split

import torch
import torchvision # for use in importing transforms
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.data.sampler import WeightedRandomSampler
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchvision import datasets
from torchvision import models
#from torchvision.transforms import ToTensor
#from torchvision.io import read_image
from PIL import Image
import random
from scipy.stats import loguniform

#from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class DanceDataset(Dataset): #custom dataset object https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.data = pd.read_csv(annotations_file, header=None) #, header=None
        self.img_labels = self.data.iloc[:,2]
        self.img_dir = self.data.iloc[:,3]
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path =self.img_dir.iloc[idx]
        image = Image.open(img_path)
        #image = read_image(img_path) #dividing by 255 here if not using ToTensor(normalizing)
        #for easier model convergence (less time taken doing dot products)
        label = self.img_labels.iloc[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

#pose labels
classes = ('Initial Posture', 'Cross Legs', 'Right Leg Up', 'Left Leg Up', 'Cross Legs Back',
        'Dancers Left Turn', 'Dancers Right Turn')

class Alex_Net(nn.Module):
    def __init__(self, num_classes: int = 7, dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# manually rewrite which CNN architecture to use (make sure to change output layer neurons)
#net = Alex_Net()
net = models.alexnet(pretrained=False)
net.classifier._modules['6'] = nn.Linear(4096, 7)
#net = models.resnet18(pretrained=False)
#net.fc = nn.Linear(512,7)

net = net.to(device)

# hyperparameters are manually set from results of hyperparameter searches
#optimizer = optim.AdamW(net.parameters(), lr = 0.0027287, betas = (0.2798, 0.64166), weight_decay = 0.00020297)
#optimizer = optim.AdamW(net.parameters(), lr = 0.000082, weight_decay = 0.001029)
optimizer = optim.SGD(net.parameters(), lr = 0.000001, momentum = 0.95165, weight_decay = 0.000154) #original model had used momentum of 0.9 and 0.0005 WD

#loss function
#criterion = nn.CrossEntropyLoss()

# Manually set if I am resizing or not. The normalization values are edited here using def get_mean_and_std
transforms = { 'Train': torchvision.transforms.Compose([
                    #transforms.RandomHorizontalFlip(p=0.4),
                    #transforms.RandomRotation(degrees=(-15, 15)),
                    #torchvision.transforms.Resize((227, 227)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.4685, 0.4762, 0.4478], [0.2007, 0.2012, 0.2038])
               ]),
               'Test': torchvision.transforms.Compose([
                    #torchvision.transforms.Resize((227, 227)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.4685, 0.4762, 0.4478], [0.2007, 0.2012, 0.2038])
               ])}

#To compile all loss results
train_loss = []
test_loss = []

#To compile all accuracy results
results_train = []
results_test = []

# To compile all precision, recall and f-measure results
precision_train_list = []
recall_train_list = []
f_measure_train_list = []

precision_test_list = []
recall_test_list = []
f_measure_test_list = []

#To compile all training/test confusian matrices per fold of last epoch, and all summed training/test confusian matrices that represent every hyperparameter choice respectively
train_conf_matrix_list_of_arrays = []
test_conf_matrix_list_of_arrays = []

layout = ['Dancers 1 and 2 and 3 (Training) - Dancers 1 and 2 and 3 (Test)']

# Iterate over the DataLoader for training data
data_train = DanceDataset("data_old_labels_dancers_1and2and3.csv", transform = transforms['Train'])

test_abs = int(len(data_train) * 0.8)
train_subset, test_subset = random_split(
    data_train, [test_abs, len(data_train) - test_abs])


#weighted cross entropy
#y = data_train.img_labels
#class_weights=class_weight.compute_class_weight('balanced',np.unique(y),y)
#class_weights=torch.tensor(class_weights,dtype=torch.float)
y1 = data_train.img_labels

class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y1), y = y1)
class_weights=torch.tensor(class_weights,dtype=torch.float) # for weighed use this instead: 

#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight = class_weights, reduction='mean')

train_loader = DataLoader(train_subset, batch_size= 8, shuffle = True)
test_loader = DataLoader(test_subset, batch_size= 8, shuffle = False)


print('-----------------------')
print(f'{layout[0]}')

for epoch in range(100):  # loop over the dataset multiple times
    
    correct = 0
    total = 0
    #For confusion matrix of train
    pred_train_labels = []
    true_train_labels = []    
    
    net.train()
    
    train_running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            ## forward + backward + optimize
            # Perform forward pass
            outputs = net(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
    
            # print statistics
            train_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            #save the predicted and true labels used per iteration
            pred_train_labels.extend(predicted)
            true_train_labels.extend(labels)
                                
    net.eval()
    
    print('---------------------------------')         
    print('...Training accuracy for epoch %d: %d %%' % (epoch + 1, 100.0 * (correct / total)))
        
    #Save the accuracy for this epoch
    results_train.append(100.0 * (correct / total))
        
    ##Confusion matrix of training data        
    #originals were a list of tensors, so this stacks them as one tensor
    pred_train_labels = torch.stack(pred_train_labels)
    true_train_labels = torch.stack(true_train_labels)
        
    cf_matrix_train = confusion_matrix(true_train_labels, pred_train_labels)
    print(cf_matrix_train)
    
    ##calculate precision, recall and f1.
    #since we have a multiclass problem, these need to be calculated for each class and then averaged
    #using the micro, macro or weighted strategies. we will be using the macro strategy since it treats
    #all classes equally. weighted strategy weighs the average in favor of classes with more samples (it
    #is like you are saying to put more emphasis on classes with more samples). micro is better for
    #balanced datasets since it works on the total TP, FP and FN unlike the others which work on per 
    #class. although, micro is important when we have observations with multiple labels assigned to
    #them which we do not (each observation has 1 label). thus, micro is similar to accuracy
    #https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
    #https://androidkt.com/micro-macro-averages-for-imbalance-multiclass-classification/
    #https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    #https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    
    #note that adding zero_division=0 is important since the predictions of the model wouldnt have classes that appear in the actual labels which gives an error
    #as such, we will have TP=0 and FP = 0 which when finding precision will give us 0/0. to avoid this, setting zero_division=0 will make all 0/0 equal to 0
    #if zero_division=1 then it will change 0/0 to 1 instead
    #https://stackoverflow.com/questions/68534836/warning-precision-and-f-score-are-ill-defined-and-being-set-to-0-0-in-labels-wi
    precision_train = precision_score(true_train_labels, pred_train_labels, average='macro', zero_division=0)
    recall_train = recall_score(true_train_labels, pred_train_labels, average='macro', zero_division=0)
    f_measure_train = 2 * (precision_train * recall_train / (precision_train + recall_train))
    
    precision_train_list.append(precision_train)
    recall_train_list.append(recall_train)
    f_measure_train_list.append(f_measure_train)
    
    print('Training Precision %.3f, Recall %.3f, F-measure %.3f' % (precision_train, recall_train, f_measure_train))
    
    correct = 0
    total = 0
    
    #for confusion matrix of test
    pred_test_labels = []
    true_test_labels = []
        
    # Iterate over the test data and generate predictions
    test_running_loss = 0.0
    with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()
                
                #save the predicted and true labels used per iteration
                pred_test_labels.extend(predicted)
                true_test_labels.extend(labels)
       
    #scheduler.step(test_running_loss/len(test_loader))
    print('-----------------------')
    print('...Test accuracy for epoch %d: %d %%' % (epoch + 1, 100.0 * (correct / total)))
        
    #Save the accuracy for this epoch
    results_test.append(100.0 * (correct / total))
        
    ##Confusion matrix of test data
    #originals were a list of tensors, so this stacks them as one tensor
    pred_test_labels = torch.stack(pred_test_labels)
    true_test_labels = torch.stack(true_test_labels)
        
    cf_matrix_test = confusion_matrix(true_test_labels, pred_test_labels)
    print(cf_matrix_test)
    
    precision_test = precision_score(true_test_labels, pred_test_labels, average='macro', zero_division=0)
    recall_test = recall_score(true_test_labels, pred_test_labels, average='macro', zero_division=0)
    f_measure_test = 2 * (precision_test * recall_test / (precision_test + recall_test))
    
    print('Test Precision %.3f, Recall %.3f, F-measure %.3f' % (precision_test, recall_test, f_measure_test))

      
    print('-----------------------')
    print('Epoch: %d, Training Loss: %.3f' % (epoch + 1, train_running_loss / len(train_loader.sampler)))
    print('Epoch: %d, Test Loss: %.3f' % (epoch + 1, test_running_loss / len(test_loader.sampler)))
    print('---------------------------------')
    
    
    #Save training and test loss for the last epoch
    train_loss.append(train_running_loss / len(train_loader.sampler))     
    test_loss.append(test_running_loss / len(test_loader.sampler))
    
    #collecting confusion matrices of last epoch for every fold (this is correct to do)
    train_conf_matrix_list_of_arrays.append(cf_matrix_train)     
    test_conf_matrix_list_of_arrays.append(cf_matrix_test)
    
    #Save test precision, recall and f-measure for the last epoch
    precision_test_list.append(precision_test)
    recall_test_list.append(recall_test)
    f_measure_test_list.append(f_measure_test)
    
    #set this as default
    if epoch == 0:
        train_loss_best = train_running_loss / len(train_loader.sampler)
        cf_matrix_best_train = cf_matrix_train
        precision_best_train = precision_train
        recall_best_train = recall_train
        f_measure_best_train = f_measure_train
    
        test_loss_best = test_running_loss / len(test_loader.sampler)
        cf_matrix_best_test = cf_matrix_test
        precision_best_test = precision_test
        recall_best_test = recall_test
        f_measure_best_test = f_measure_test
    
    f_measure_test_this_epoch = f_measure_test
    #Save the best confusion matrix + performance metrics with the highest test f1 measure
    if f_measure_test_this_epoch > f_measure_best_test:
        train_loss_best = train_running_loss / len(train_loader.sampler)
        cf_matrix_best_train = cf_matrix_train
        precision_best_train = precision_train
        recall_best_train = recall_train
        f_measure_best_train = f_measure_train
        
        test_loss_best = test_running_loss / len(test_loader.sampler)
        cf_matrix_best_test = cf_matrix_test
        precision_best_test = precision_test
        recall_best_test = recall_test
        f_measure_best_test = f_measure_test
        
        print("Performance metrics for best epoch updated!")
        print('---------------------------------')