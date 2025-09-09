## CHAPTER 4 APPLICATION

import pandas as pd
#import torch
import numpy as np
#import tensorly as tl
#import sys
import random
from torch.utils.data import Dataset
#from PIL import Image
import cv2
#import torchvision # for use in importing transforms [CURRENTLY NOT SUPPORT WITH NUMPY 2.0]
from sklearn import preprocessing
from functools import reduce # for the outer product of multiple vectors calculation
#import math # for natural logarithm (USED NUMPY INSTEAD SINCE IT DOESNT OVERFLOW)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
#from sklearn.utils import class_weight
#from torch import logsumexp
from scipy.special import logsumexp
from scipy.stats import loguniform
from sklearn.model_selection import StratifiedShuffleSplit
import math
import cpdecomp_loglikelihood_MultiProcess_py # custom-made .py file containing a definition. This was done since there was an issue with getting multiprocessing to work and importing them like this fixed it
import derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py # custom-made .py file containing a definition. This was done since there was an issue with getting multiprocessing to work and importing them like this fixed it
import predicted_labels_MultiProcess_py # custom-made .py file containing a definition. This was done since there was an issue with getting multiprocessing to work and importing them like this fixed it
import concurrent.futures as cf # IMPORTED AFTER I HAVE IMPORTED THE CUSTOM-MADE .py FILES
import itertools
import time
#import os
#import warnings

## SETTING THE SEED FOR REPRODUCIBLE RESULTS
#random.seed(a = 5)

## Ignore all errors (a bunch of overflow errors might come up from dividing by np.exp(ln_of_innerproduct_exponentials_list) but these will return 0 anyway)
#warnings.filterwarnings("ignore", category = RuntimeWarning)

## CUSTOM DATASET OBJECT FOR READING THE DATA FILE CONTAINING BOTH THE IMAGES AND LABELS
class DanceDataset(Dataset): #custom dataset object https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __init__(self, annotations_file): #, transform=None, target_transform=None
        self.data = pd.read_csv(annotations_file, header=None) #, header=None
        self.img_labels = self.data.iloc[:,2]
        self.img_dir = self.data.iloc[:,3]
        
        #self.transform = transform
        #self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path =self.img_dir.iloc[idx]
        #image = Image.open(img_path)
        #image = read_image(img_path) / 255 #dividing by 255 here if not using ToTensor(normalizing)
        
        image = cv2.imread(img_path)   # reads an image in the BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        image = image.astype(np.float32)
        mean, std_dev = cv2.meanStdDev(image)
        image[:,:,0] = (image[:,:,0] - mean[0]) / std_dev[0]
        image[:,:,1] = (image[:,:,1] - mean[1]) / std_dev[1]
        image[:,:,2] = (image[:,:,2] - mean[2]) / std_dev[2]
        
        label = self.img_labels.iloc[idx]
        
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
            
        return image, label


## TRANSFORMING IMAGE TO TENSOR DATA (MAYBE NORMALIZE?)
#transforms = { 'Train': torchvision.transforms.Compose([
#                    #transforms.RandomHorizontalFlip(p=0.4),
#                    #transforms.RandomRotation(degrees=(-15, 15)),
#                    #torchvision.transforms.Resize((227, 227)),
#                    torchvision.transforms.ToTensor(),
#                    #torchvision.transforms.Normalize([0.4770, 0.4807, 0.4548], [0.2045, 0.2034, 0.2070])
#               ]),
#               'Test': torchvision.transforms.Compose([
#                    #torchvision.transforms.Resize((227, 227)),
#                    torchvision.transforms.ToTensor(),
#                    #torchvision.transforms.Normalize([0.4770, 0.4807, 0.4548], [0.2045, 0.2034, 0.2070])
#               ])}

### LOADING THE DATA
## Training using 2 dancers and testing on the other
#data_train_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancers_1and2.csv", transform = transforms['Train'])
#data_test_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancer_3.csv", transform = transforms['Test'])
data_train_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancers_1and3.csv")
data_test_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancer_2.csv")

data_tensors_train = []
true_labels_train = []

for i in range(len(data_train_customdataobject)):
        # get the inputs; custom data object is a list of [inputs, labels] that need to be called to get them
        inputs, labels = data_train_customdataobject[i]
        inputs = np.array(inputs)
        # save the inputs; get the tensor and label for each entry and save them as lists
        data_tensors_train.append(inputs) # list of tensors
        true_labels_train.append(labels)
        
data_tensors_test = []
true_labels_test = []

for i in range(len(data_test_customdataobject)):
        # get the inputs; custom data object is a list of [inputs, labels] that need to be called to get them
        inputs, labels = data_test_customdataobject[i]
        #inputs = data_test_customdataobject[i]
        inputs = np.array(inputs)
        # save the inputs; get the tensor and label for each entry and save them as lists
        data_tensors_test.append(inputs) # list of tensors
        true_labels_test.append(labels)



## Training and testing using 3 dancers
#data_train_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancers_1and2and3.csv", transform = transforms['Train'])
data_train_customdataobject = DanceDataset("C:/Users/Andrew/Downloads/Masters Data/PostureRecognition - The third folder/data_old_labels_dancers_1and2and3.csv")

data_tensors = []
true_labels = []

for i in range(len(data_train_customdataobject)):
        # get the inputs; custom data object is a list of [inputs, labels] that need to be called to get them
        inputs, labels = data_train_customdataobject[i]
        inputs = np.array(inputs)
        # save the inputs; get the tensor and label for each entry and save them as lists
        data_tensors.append(inputs) # list of tensors
        true_labels.append(labels)

# taking a stratified sample of the training and test sets
sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 5)

# get train and test indices
for i, (train_index, test_index) in enumerate(sss.split(data_tensors, true_labels)):
     print(f"Fold {i}:")
     print(f"  Train: index={train_index}")
     print(f"  Test:  index={test_index}")

# convert from array type to list type
train_index = train_index.tolist()
test_index = test_index.tolist()

data_tensors_train = []
true_labels_train = []

for i in range(len(train_index)):
        # get the inputs; custom data object is a list of [inputs, labels] that need to be called to get them
        inputs = data_tensors[train_index[i]]
        labels = true_labels[train_index[i]]
        
        # save the inputs; get the tensor and label for each entry and save them as lists
        data_tensors_train.append(inputs) # list of tensors
        true_labels_train.append(labels)

data_tensors_test = []
true_labels_test = []

for i in range(len(test_index)):
        # get the inputs; custom data object is a list of [inputs, labels] that need to be called to get them
        inputs = data_tensors[test_index[i]]
        labels = true_labels[test_index[i]]
        
        # save the inputs; get the tensor and label for each entry and save them as lists
        data_tensors_test.append(inputs) # list of tensors
        true_labels_test.append(labels)

# checking if the strata are correct
# import collections
# counter1 = collections.Counter(true_labels_train)
# counter2 = collections.Counter(true_labels_test)

# gives a sorted order for the labels
# [counter1[x] for x in sorted(counter1.keys())]
# [counter2[x] for x in sorted(counter2.keys())]


### ONE HOT ENCODING THE TRUE LABELS

array_img_labels_train = np.array(true_labels_train)
array_img_labels_test = np.array(true_labels_test)

OneHotEncoder = preprocessing.OneHotEncoder(sparse_output = False)

# reshaping the array so that it can work with the encoder 
array_img_labels_train = array_img_labels_train.reshape(len(array_img_labels_train),1)
array_img_labels_test = array_img_labels_test.reshape(len(array_img_labels_test),1)

img_labels_train_onehotencoded = OneHotEncoder.fit_transform(array_img_labels_train)
img_labels_test_onehotencoded = OneHotEncoder.fit_transform(array_img_labels_test)

# Note that I checked and correctly found that the first column of the onehotencoded matrix represents the first label, the second column for the second label, etc



## TAKING MINI-BATCHES OF THE TRAINING SET
# Define a list of batch sizes which we will be taking a random value of each time
L = [2, 4, 8, 16, 32, 64, 128, 256]

batch_size = random.choice(L)

# defining a list of values up to the number of samples in the training data so that we can randomize it
number_of_objects = list(range(len(data_tensors_train)))
random.shuffle(number_of_objects)

# define empty list and array to fill them with the randomized data instead
data_tensors_train_random = [0] * len(data_tensors_train)
img_labels_train_onehotencoded_random = np.zeros((np.shape(img_labels_train_onehotencoded)[0], np.shape(img_labels_train_onehotencoded)[1]))
# ^ np.shape(img_labels_train_onehotencoded)[0] to extract number of rows in img_labels_train_onehotencoded and np.shape(img_labels_train_onehotencoded)[1] to extract number of columns in img_labels_train_onehotencoded

for i in range(len(number_of_objects)):
    data_tensors_train_random[i] = data_tensors_train[number_of_objects[i]]
    img_labels_train_onehotencoded_random[i,:] = img_labels_train_onehotencoded[number_of_objects[i],:]


# create duplicates of training data and labels which we will be removing elements of in the for loop
data_tensors_train_foriter = data_tensors_train_random
true_labels_train_foriter = img_labels_train_onehotencoded_random

# create empty lists of zeros proportional to the amount of batches that we will have
# each list will become a lists of lists containing each batch
data_tensors_train_random_batches = [0] * math.ceil(len(data_tensors_train) / batch_size)
img_labels_train_onehotencoded_random_batches = [0] * math.ceil(len(data_tensors_train) / batch_size)

for i in range(len(data_tensors_train_random_batches)):
    # define each respective batch
    data_tensors_train_random_batches[i] = list(data_tensors_train_foriter[:batch_size])
    img_labels_train_onehotencoded_random_batches[i] = list(true_labels_train_foriter[:batch_size,:])
    
    # remove said batch from the list to not be included again
    data_tensors_train_foriter = data_tensors_train_foriter[batch_size:]
    true_labels_train_foriter = true_labels_train_foriter[batch_size:,:]


## DEFINING THE RANK OF THE TENSOR DECOMPOSITION

# ONE OPTION IS TO TAKE RANDOM VALUES FROM A UNIDORM DISTRIBUTION BETWEEN -1 AND 1 FOR WEIGHT INITIALIZATION OF THE FACTOR MATRICES AND BIAS
# TO TRY TO REDUCE THE SIZES OF THE INITILIAZED WEIGHTS, I'M FOLLOWING A GENERAL METHOD OF INITIALIZATION FROM HERE: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
# ^KNOWN AS 'Kaiming He initialization': TAKING NORMAL DISTRIBUTION OF MEAN 0 AND STANDARD DEVIATION 1/SQRT(n), WHERE n IS THE AMOUNT OF TRAINING SAMPLE. ON UNIFORM BETWEEN -1 AND 1 AND STANDARD NORMAL I WAS GETTING CONSTANT OVERFLOW ERRORS, SO WANT TO SEE IF THIS HELPS
# TAKING RANK OF CP DECOMPOSITION AS FOLLOWS:
cpdecomp_rank = 3
# --- If changing make sure to update vectorize_tensor_3D and the mini-batch parts ---

# Define the initial bias vector
bias_cpdecomp_LIST = [np.random.normal(0, 1 / math.sqrt(len(data_tensors_train))) for _ in range(6)]

# Initialize weights for the factor matrices
factormatrix_cpdecomp_1_CLASS1 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS1 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS1 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS2 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS2 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS2 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS3 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS3 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS3 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS4 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS4 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS4 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS5 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS5 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS5 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS6 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (240,cpdecomp_rank)) #np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS6 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (320,cpdecomp_rank)) #np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS6 = np.random.normal(0, 1 / math.sqrt(len(data_tensors_train)), (3,cpdecomp_rank)) #np.random.uniform(-1,1, (3,cpdecomp_rank))

# Initialize weights for the factor matrices (trying to get that overflow error again)
bias_cpdecomp_LIST = [random.uniform(-1,1) for _ in range(6)]

factormatrix_cpdecomp_1_CLASS1 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS1 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS1 = np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS2 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS2 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS2 = np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS3 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS3 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS3 = np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS4 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS4 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS4 = np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS5 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS5 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS5 = np.random.uniform(-1,1, (3,cpdecomp_rank))

factormatrix_cpdecomp_1_CLASS6 = np.random.uniform(-1,1, (240,cpdecomp_rank))
factormatrix_cpdecomp_2_CLASS6 = np.random.uniform(-1,1, (320,cpdecomp_rank))
factormatrix_cpdecomp_3_CLASS6 = np.random.uniform(-1,1, (3,cpdecomp_rank))


## Tucker Decomposition (Unused)
tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3 = 3, 4, 5

bias_tuckerdecomp_CLASS1 = random.uniform(-1, 1)
bias_tuckerdecomp_CLASS2 = random.uniform(-1, 1)
bias_tuckerdecomp_CLASS3 = random.uniform(-1, 1)
bias_tuckerdecomp_CLASS4 = random.uniform(-1, 1)
bias_tuckerdecomp_CLASS5 = random.uniform(-1, 1)
bias_tuckerdecomp_CLASS6 = random.uniform(-1, 1)

bias_tuckerdecomp_LIST = [bias_tuckerdecomp_CLASS1, bias_tuckerdecomp_CLASS2, bias_tuckerdecomp_CLASS3, bias_tuckerdecomp_CLASS4, bias_tuckerdecomp_CLASS5, bias_tuckerdecomp_CLASS6]

corematrix_tuckerdecomp_CLASS1 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS1 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS1 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS1 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))

corematrix_tuckerdecomp_CLASS2 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS2 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS2 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS2 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))

corematrix_tuckerdecomp_CLASS3 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS3 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS3 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS3 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))

corematrix_tuckerdecomp_CLASS4 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS4 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS4 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS4 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))

corematrix_tuckerdecomp_CLASS5 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS5 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS5 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS5 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))

corematrix_tuckerdecomp_CLASS6 = np.random.uniform(-1,1, (tuckerdecomp_rank_1,tuckerdecomp_rank_2, tuckerdecomp_rank_3))
factormatrix_tuckerdecomp_1_CLASS6 = np.random.uniform(-1,1, (240,tuckerdecomp_rank_1))
factormatrix_tuckerdecomp_2_CLASS6 = np.random.uniform(-1,1, (320,tuckerdecomp_rank_2))
factormatrix_tuckerdecomp_3_CLASS6 = np.random.uniform(-1,1, (3,tuckerdecomp_rank_3))


## OUTER PRODUCT OF N VECTORS
# gotten from https://stackoverflow.com/questions/17138393/numpy-outer-product-of-n-vectors
#def outer1(*vs): -> UNUSED NOT AS EFFECTIVE
#    return np.multiply.reduce(np.ix_(*vs))

def outer2(*vs):
    return reduce(np.multiply.outer, vs)
# ^ using the above since it worked with correctly with example u = [1, -2, 0] and v = [2, -2, 3, -6] in outer2(u,v)
# WORKED CORRECTLY WHEN I ADDED A CONSTANT WITH THE VECTORS, e.g. taking y = [6,7,9,2] and z = 5 in outer2(z,u,v,y)

## VECTORIZE A 3D TENSOR

# manually made, compared with vectorize_tensor_3D_old, this is a numpy only alternative and comparing the results from both gives the same outputs
def vectorize_tensor_3D(tensor):
    # Define the channels as separate matrices (to be updated in case of changed tensor decomposition rank)
    tensorarray_1st = tensor[:,:,0]
    tensorarray_2nd = tensor[:,:,1]
    tensorarray_3rd = tensor[:,:,2]
    
    # Flatten them in column-major order
    tensorvector_1st = tensorarray_1st.flatten('F')
    tensorvector_2nd = tensorarray_2nd.flatten('F')
    tensorvector_3rd = tensorarray_3rd.flatten('F')
    
    # Stack them as one vector
    return np.hstack((tensorvector_1st, tensorvector_2nd, tensorvector_3rd))

# BLOCK LOG-LIKELIHOOD CALCULATION
def log_likelihood_innerproducts(data_tensors_train_random_batches, minibatch, outer_product_factor_matrices_class1_rank1, outer_product_factor_matrices_class1_rank2, outer_product_factor_matrices_class1_rank3, outer_product_factor_matrices_class2_rank1, outer_product_factor_matrices_class2_rank2, outer_product_factor_matrices_class2_rank3,
        outer_product_factor_matrices_class3_rank1, outer_product_factor_matrices_class3_rank2, outer_product_factor_matrices_class3_rank3, outer_product_factor_matrices_class4_rank1, outer_product_factor_matrices_class4_rank2, outer_product_factor_matrices_class4_rank3,
        outer_product_factor_matrices_class5_rank1, outer_product_factor_matrices_class5_rank2, outer_product_factor_matrices_class5_rank3, outer_product_factor_matrices_class6_rank1, outer_product_factor_matrices_class6_rank2, outer_product_factor_matrices_class6_rank3):
    
    inner_product_class1_list = []
    inner_product_class2_list = []
    inner_product_class3_list = []
    inner_product_class4_list = []
    inner_product_class5_list = []
    inner_product_class6_list = []
    ln_of_innerproduct_exponentials_list = []
    log_likelihood_list = []
    
    for i in range(len(data_tensors_train_random_batches[minibatch])):
        inner_product_class1 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class1_rank1 + outer_product_factor_matrices_class1_rank2 + outer_product_factor_matrices_class1_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        inner_product_class2 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class2_rank1 + outer_product_factor_matrices_class2_rank2 + outer_product_factor_matrices_class2_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        inner_product_class3 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class3_rank1 + outer_product_factor_matrices_class3_rank2 + outer_product_factor_matrices_class3_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        inner_product_class4 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class4_rank1 + outer_product_factor_matrices_class4_rank2 + outer_product_factor_matrices_class4_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        inner_product_class5 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class5_rank1 + outer_product_factor_matrices_class5_rank2 + outer_product_factor_matrices_class5_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        inner_product_class6 = np.inner(vectorize_tensor_3D(outer_product_factor_matrices_class6_rank1 + outer_product_factor_matrices_class6_rank2 + outer_product_factor_matrices_class6_rank3), vectorize_tensor_3D(data_tensors_train_random_batches[minibatch][i]))
        
        # Calculate the other logarithmic term needed in the log-likelihood (CAN'T BE USED SINCE IT LEAD TO OVERFLOWS IN EXPONENTIALS)
        ln_of_innerproduct_exponentials = np.log(1 + np.exp(bias_cpdecomp_LIST[0] + inner_product_class1) + np.exp(bias_cpdecomp_LIST[1] + inner_product_class2) + np.exp(bias_cpdecomp_LIST[2] + inner_product_class3) + np.exp(bias_cpdecomp_LIST[3] + inner_product_class4) + np.exp(bias_cpdecomp_LIST[4] + inner_product_class5) + np.exp(bias_cpdecomp_LIST[5] + inner_product_class6))

        # TO CALCULATE ln_of_innerproduct_exponentials I'LL BE DOING A ROUNDABOUT METHOD. TESTED IT FOR SMALL VALUES DEFINED IN AND WORKED CORRECTLY. NOT SURE FOR LARGER VALUES IF FINE BUT GOOD ENOUGH
        # GOT IDEA FROM https://stackoverflow.com/questions/44033533/how-to-deal-with-exponent-overflow-of-64float-precision-in-python , https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html AND https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp
        
        ln_of_innerproduct_exponentials = logsumexp([[0], [bias_cpdecomp_LIST[0] + inner_product_class1], [bias_cpdecomp_LIST[1] + inner_product_class2], [bias_cpdecomp_LIST[2] + inner_product_class3], [bias_cpdecomp_LIST[3] + inner_product_class4], [bias_cpdecomp_LIST[4] + inner_product_class5], [bias_cpdecomp_LIST[5] + inner_product_class6]])
        
        # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
        #temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
        #ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
        #ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]

        # log-likelihood per sample i
        log_likelihood = img_labels_train_onehotencoded_random_batches[minibatch][i][0]*(bias_cpdecomp_LIST[0] + inner_product_class1 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][1]*(bias_cpdecomp_LIST[1] + inner_product_class2 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][2]*(bias_cpdecomp_LIST[2] + inner_product_class3 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][3]*(bias_cpdecomp_LIST[3] + inner_product_class4 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][4]*(bias_cpdecomp_LIST[4] + inner_product_class5 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][5]*(bias_cpdecomp_LIST[5] + inner_product_class6 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][6]*(0 - ln_of_innerproduct_exponentials)
        
        inner_product_class1_list.append(inner_product_class1)
        inner_product_class2_list.append(inner_product_class2)
        inner_product_class3_list.append(inner_product_class3)
        inner_product_class4_list.append(inner_product_class4)
        inner_product_class5_list.append(inner_product_class5)
        inner_product_class6_list.append(inner_product_class6)
        ln_of_innerproduct_exponentials_list.append(ln_of_innerproduct_exponentials)
        log_likelihood_list.append(log_likelihood)
        
    log_likelihood_total = sum(log_likelihood_list)
        
    return log_likelihood_total, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list


# manually made, works good on an example tensor
#def vectorize_tensor_3D_old(tensor):
#    
#    tensor_dimensions = tensor.shape # returns a tuple with the dimensions of the tensor
#    # ^ THE FIRST DIMENSION IS THE NUMBER OF CHANNELS, THE SECOND IS THE NUMBER OF ROWS AND THE THIRD IS THE NUMBER OF COLUMNS
#    vectorized_tensor = []
#    
#    for j in range(tensor_dimensions[0]):
#        for k in range(tensor_dimensions[2]):
#                
#            vec = tensor[j,:,k]
#                
#            vectorized_tensor.extend(vec)
#
#    return np.array(vectorized_tensor)


## TAKING INNER PRODUCT OF TWO TENSORS

# FIRST VECTORIZE EACH TENSOR (using 'vectorize_tensor_3D' above) AND THEN TAKE THEIR INNER PRODUCT USING np.inner()


## LOG-LIKELIHOODS

# (Unused)
def tuckerdecomp_loglikelihood(data, labels_onehotencoded, tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3, bias_vector, corematrix_tuckerdecomp_CLASS1, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, corematrix_tuckerdecomp_CLASS2, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, corematrix_tuckerdecomp_CLASS3, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
                           , corematrix_tuckerdecomp_CLASS4, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, corematrix_tuckerdecomp_CLASS5, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, corematrix_tuckerdecomp_CLASS6, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6):
    
    log_likelihood_list = []
    inner_product_class1_list = []
    inner_product_class2_list = []
    inner_product_class3_list = []
    inner_product_class4_list = []
    inner_product_class5_list = []
    inner_product_class6_list = []
    ln_of_innerproduct_exponentials_list = []
    
    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
    outer_product_core_factor_matrices_class1_list = []
    outer_product_core_factor_matrices_class2_list = []
    outer_product_core_factor_matrices_class3_list = []
    outer_product_core_factor_matrices_class4_list = []
    outer_product_core_factor_matrices_class5_list = []
    outer_product_core_factor_matrices_class6_list = []
    
    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
    for x in range(tuckerdecomp_rank_1):
        for y in range(tuckerdecomp_rank_2):
            for z in range(tuckerdecomp_rank_3):
                outer_product_core_factor_matrices_class1 = outer2(corematrix_tuckerdecomp_CLASS1[z,x,y], factor_matrix1_class1[:,x], factor_matrix2_class1[:,y], factor_matrix3_class1[:,z])
                outer_product_core_factor_matrices_class2 = outer2(corematrix_tuckerdecomp_CLASS2[z,x,y], factor_matrix1_class2[:,x], factor_matrix2_class2[:,y], factor_matrix3_class2[:,z])
                outer_product_core_factor_matrices_class3 = outer2(corematrix_tuckerdecomp_CLASS3[z,x,y], factor_matrix1_class3[:,x], factor_matrix2_class3[:,y], factor_matrix3_class3[:,z])
                outer_product_core_factor_matrices_class4 = outer2(corematrix_tuckerdecomp_CLASS4[z,x,y], factor_matrix1_class4[:,x], factor_matrix2_class4[:,y], factor_matrix3_class4[:,z])
                outer_product_core_factor_matrices_class5 = outer2(corematrix_tuckerdecomp_CLASS5[z,x,y], factor_matrix1_class5[:,x], factor_matrix2_class5[:,y], factor_matrix3_class5[:,z])
                outer_product_core_factor_matrices_class6 = outer2(corematrix_tuckerdecomp_CLASS6[z,x,y], factor_matrix1_class6[:,x], factor_matrix2_class6[:,y], factor_matrix3_class6[:,z])
                
                outer_product_core_factor_matrices_class1_list.append(outer_product_core_factor_matrices_class1)
                outer_product_core_factor_matrices_class2_list.append(outer_product_core_factor_matrices_class2)
                outer_product_core_factor_matrices_class3_list.append(outer_product_core_factor_matrices_class3)
                outer_product_core_factor_matrices_class4_list.append(outer_product_core_factor_matrices_class4)
                outer_product_core_factor_matrices_class5_list.append(outer_product_core_factor_matrices_class5)
                outer_product_core_factor_matrices_class6_list.append(outer_product_core_factor_matrices_class6)
    
    for i in range(len(labels_onehotencoded)):
        # ^ getting sample size from that
        
        # Sum all them up in the list and then take the inner product of them with the ith data
        inner_product_class1 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class1_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class2 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class2_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class3 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class3_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class4 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class4_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class5 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class5_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class6 = np.inner(vectorize_tensor_3D(np.sum(outer_product_core_factor_matrices_class6_list, axis = 0)), vectorize_tensor_3D(data[i]))
        
        # Calculate the other logarithmic term needed in the log-likelihood (CAN'T BE USED SINCE IT LEAD TO OVERFLOWS IN EXPONENTIALS)
        #ln_of_innerproduct_exponentials = np.log(1 + np.exp(bias_vector[0] + inner_product_class1) + np.exp(bias_vector[1] + inner_product_class2) + np.exp(bias_vector[2] + inner_product_class3) + np.exp(bias_vector[3] + inner_product_class4) + np.exp(bias_vector[4] + inner_product_class5) + np.exp(bias_vector[5] + inner_product_class6))
        
        # TO CALCULATE ln_of_innerproduct_exponentials I'LL BE DOING A ROUNDABOUT METHOD. TESTED IT FOR SMALL VALUES DEFINED IN AND WORKED CORRECTLY. NOT SURE FOR LARGER VALUES IF FINE BUT GOOD ENOUGH
        # GOT IDEA FROM https://stackoverflow.com/questions/44033533/how-to-deal-with-exponent-overflow-of-64float-precision-in-python , https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html AND https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp
        
        ln_of_innerproduct_exponentials = logsumexp([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]])
        
        # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
        #temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
        #ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
        #ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]
        
        # log-likelihood per sample i
        log_likelihood = labels_onehotencoded[i][0]*(bias_vector[0] + inner_product_class1 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][1]*(bias_vector[1] + inner_product_class2 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][2]*(bias_vector[2] + inner_product_class3 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][3]*(bias_vector[3] + inner_product_class4 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][4]*(bias_vector[4] + inner_product_class5 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][5]*(bias_vector[5] + inner_product_class6 - ln_of_innerproduct_exponentials) + labels_onehotencoded[i][6]*(0 - ln_of_innerproduct_exponentials)
        
        # saving the ith log-likelihood in this list to sum all of them later
        log_likelihood_list.append(log_likelihood)
        
        # also saving the inner_product_classes and ln_of_innerproduct_exponentials of each iteration
        inner_product_class1_list.append(inner_product_class1)
        inner_product_class2_list.append(inner_product_class2)
        inner_product_class3_list.append(inner_product_class3)
        inner_product_class4_list.append(inner_product_class4)
        inner_product_class5_list.append(inner_product_class5)
        inner_product_class6_list.append(inner_product_class6)
        ln_of_innerproduct_exponentials_list.append(ln_of_innerproduct_exponentials)
    
    log_likelihood_total = sum(log_likelihood_list)
    
    return log_likelihood_total, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list


## LOG-LIKELIHOOD DERIVATIVES

# -- USING CP DECOMPOSITION --
# BIAS OF EVERY CLASS
def derivative_cportuckerdecomp_wrt_bias_vector(data, labels_onehotencoded, bias_vector, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    derivative_log_likelihood_bias_vector_list = []
    
    # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt bias of ANY CLASS r. The bias vector contains all values of each class
    # term_1 is always 1
    term_1 = 1
    
    for i in range(len(labels_onehotencoded)):
        # define an empty vector which will contain all values of the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
        derivative_log_likelihood_bias_vector_ithdata = np.zeros(shape = 6)
        
        for k in range(6):
            if k == 0:
                
                term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][0]*(term_1) - term_2 
            
            elif k == 1:
                term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][1]*(term_1) - term_2
            
            elif k == 2:
                term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][2]*(term_1) - term_2
            
            elif k == 3:
                term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][3]*(term_1) - term_2
            
            elif k == 4:
                term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][4]*(term_1) - term_2
        
            else:
                term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                
                if math.isnan(term_2):
                    term_2 = 0
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood_bias_vector_ithdata[k] = labels_onehotencoded[i][5]*(term_1) - term_2
                
        derivative_log_likelihood_bias_vector_list.append(derivative_log_likelihood_bias_vector_ithdata)
    
    derivative_log_likelihood_bias_vector = np.sum(derivative_log_likelihood_bias_vector_list, axis = 0)
    
    return derivative_log_likelihood_bias_vector


## -- USING TUCKER DECOMPOSITION -- (Unused)
# FACTOR MATRIX 1 OF ANY CLASS [not used]
def derivative_tuckerdecomp_wrt_weight_factor_matrix1(data, class_12345or6, labels_onehotencoded, tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3, bias_vector, corematrix_tuckerdecomp_CLASS12345or6, factormatrix2_class_12345or6, factormatrix3_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    derivative_log_likelihood_weight_factormatrix1_list = []
    
    for i in range(len(labels_onehotencoded)):
        # define an empty matrix which will contain all values of term 1 for the partial derivative of log-likelihood wrt factor matrix 1 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_1}, for r = 1, ..., 6
        # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt factor matrix 1 of ANY CLASS r
        #term_1_weight_factormatrix1 = np.zeros(shape = (240, tuckerdecomp_rank_1)) UNNECESSARY
        
        # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{1, tuckerdecomp_rank_1, x} where this will be done for each tuckerdecomp_rank_1 = 1, ..., P_1 and x = 1, ..., 240
        term_1_needtosum_list = []
        
        # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 1 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_1}
        derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, tuckerdecomp_rank_1))
        
        for j1 in range(tuckerdecomp_rank_1):
            for x in range(240):
                for j2 in range(tuckerdecomp_rank_2):
                    for j3 in range(tuckerdecomp_rank_3):
                        for y in range(320):
                            for z in range(3):
                                # We differentiated  
                                term_1_needtosum = data[i][z, x, y] * corematrix_tuckerdecomp_CLASS12345or6[j3, j1, j2] * factormatrix2_class_12345or6[y, j2] * factormatrix3_class_12345or6[z, j3]
                    
                                term_1_needtosum_list.append(term_1_needtosum)
            
                term_1 = sum(term_1_needtosum_list)
                
                #term_1_weight_factormatrix1[x, j1] = term_1
                
                # refresh this back to empty
                term_1_needtosum_list = []
                
                # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt factor matrix 1 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_1}
                # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt factor matrix 1 of ANY CLASS
                #term_2_weight_factormatrix1 = np.zeros(shape = (240, tuckerdecomp_rank_1)) UNNECESSARY
                
                
                if class_12345or6 == 1:
                    term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(term_1 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
                elif class_12345or6 == 2:
                    term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(term_1 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
                elif class_12345or6 == 3:
                    term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(term_1 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
                elif class_12345or6 == 4:
                    term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(term_1 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
                elif class_12345or6 == 5:
                    term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(term_1 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
                else:
                    term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix1[x, j1] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(term_1 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix1_ithdata[x, j1] = derivative_log_likelihood
                
        derivative_log_likelihood_weight_factormatrix1_list.append(derivative_log_likelihood_weight_factormatrix1_ithdata)
            
    
    derivative_log_likelihood_weight_factormatrix1 = sum(derivative_log_likelihood_weight_factormatrix1_list)
    
    return derivative_log_likelihood_weight_factormatrix1

# FACTOR MATRIX 2 OF ANY CLASS [not used]
def derivative_tuckerdecomp_wrt_weight_factor_matrix2(data, class_12345or6, labels_onehotencoded, tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3, bias_vector, corematrix_tuckerdecomp_CLASS12345or6, factormatrix1_class_12345or6, factormatrix3_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    derivative_log_likelihood_weight_factormatrix2_list = []
    
    for i in range(len(labels_onehotencoded)):
        # define an empty matrix which will contain all values of term 1 for the partial derivative of log-likelihood wrt factor matrix 2 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_2} for r = 1, ..., 6
        # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt factor matrix 2 of ANY CLASS r
        #term_1_weight_factormatrix2 = np.zeros(shape = (320, tuckerdecomp_rank_2)) UNNECESSARY
        
        # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{2, tuckerdecomp_rank_2, y} where this will be done for each tuckerdecomp_rank_2 = 1, ..., P_2 and y = 1, ..., 320
        term_1_needtosum_list = []
        
        # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 2 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_2}
        derivative_log_likelihood_weight_factormatrix2_ithdata = np.zeros(shape = (320, tuckerdecomp_rank_2))
        
        for j2 in range(tuckerdecomp_rank_2):
            for y in range (320):
                for j1 in range(tuckerdecomp_rank_1):
                    for j3 in range(tuckerdecomp_rank_3):
                        for x in range(240):
                            for z in range(3):
                                # We differentiated  
                                term_1_needtosum = data[i][z, x, y] * corematrix_tuckerdecomp_CLASS12345or6[j3, j1, j2] * factormatrix1_class_12345or6[x, j1] * factormatrix3_class_12345or6[z, j3]
                    
                                term_1_needtosum_list.append(term_1_needtosum)
                
                term_1 = sum(term_1_needtosum_list)
                
                #term_1_weight_factormatrix2[y, j2] = term_1
                
                # refresh this back to empty
                term_1_needtosum_list = []
                
                # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt factor matrix 2 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_2}
                # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt factor matrix 2 of ANY CLASS
                #term_2_weight_factormatrix2 = np.zeros(shape = (320, tuckerdecomp_rank_2)) UNNECESSARY
                
                
                if class_12345or6 == 1:
                    term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(term_1 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
                elif class_12345or6 == 2:
                    term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(term_1 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
                elif class_12345or6 == 3:
                    term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(term_1 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
                elif class_12345or6 == 4:
                    term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(term_1 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
                elif class_12345or6 == 5:
                    term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(term_1 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
                else:
                    term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix2[y, j2] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(term_1 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix2_ithdata[y, j2] = derivative_log_likelihood
                
        derivative_log_likelihood_weight_factormatrix2_list.append(derivative_log_likelihood_weight_factormatrix2_ithdata)
            
    
    derivative_log_likelihood_weight_factormatrix2 = sum(derivative_log_likelihood_weight_factormatrix2_list)
    
    return derivative_log_likelihood_weight_factormatrix2

# FACTOR MATRIX 3 OF ANY CLASS [not used]
def derivative_tuckerdecomp_wrt_weight_factor_matrix3(data, class_12345or6, labels_onehotencoded, tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3, bias_vector, corematrix_tuckerdecomp_CLASS12345or6, factormatrix1_class_12345or6, factormatrix2_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    derivative_log_likelihood_weight_factormatrix3_list = []
    
    for i in range(len(labels_onehotencoded)):
        # define an empty matrix which will contain all values of term 1 for the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3} for r = 1, ..., 6
        # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt factor matrix 3 of ANY CLASS r
        #term_1_weight_factormatrix3 = np.zeros(shape = (3, tuckerdecomp_rank_3)) UNNECESSARY
        
        # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{3, tuckerdecomp_rank_3, z} where this will be done for each tuckerdecomp_rank_3 = 1, ..., P_3 and z = 1, 2, 3
        term_1_needtosum_list = []
        
        # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
        derivative_log_likelihood_weight_factormatrix3_ithdata = np.zeros(shape = (3, tuckerdecomp_rank_3))
        
        for j3 in range(tuckerdecomp_rank_3):
            for z in range(3):
                for j1 in range(tuckerdecomp_rank_1):
                    for j2 in range(tuckerdecomp_rank_2):
                        for x in range(240):
                            for y in range(320):
                                # We differentiated  
                                term_1_needtosum = data[i][z, x, y] * corematrix_tuckerdecomp_CLASS12345or6[j3, j1, j2] * factormatrix1_class_12345or6[x, j1] * factormatrix2_class_12345or6[y, j2]
                    
                                term_1_needtosum_list.append(term_1_needtosum)
            
                term_1 = sum(term_1_needtosum_list)
                
                #term_1_weight_factormatrix3[z, j3] = term_1
                
                # refresh this back to empty
                term_1_needtosum_list = []
                
                # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
                # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt factor matrix 3 of ANY CLASS
                #term_2_weight_factormatrix3 = np.zeros(shape = (3, tuckerdecomp_rank_3)) UNNECESSARY
                
                
                if class_12345or6 == 1:
                    term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(term_1 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
                elif class_12345or6 == 2:
                    term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(term_1 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
                elif class_12345or6 == 3:
                    term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(term_1 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
                elif class_12345or6 == 4:
                    term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(term_1 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
                elif class_12345or6 == 5:
                    term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(term_1 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
                else:
                    term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                    
                    #term_2_weight_factormatrix3[z, j3] = term_2
                    
                    # Calculate the partial derivative of log-likelihood wrt weight value
                    derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(term_1 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                    
                    derivative_log_likelihood_weight_factormatrix3_ithdata[z, j3] = derivative_log_likelihood
                
        derivative_log_likelihood_weight_factormatrix3_list.append(derivative_log_likelihood_weight_factormatrix3_ithdata)
            
    
    derivative_log_likelihood_weight_factormatrix3 = sum(derivative_log_likelihood_weight_factormatrix3_list)
    
    return derivative_log_likelihood_weight_factormatrix3


# CORE MATRIX OF ANY CLASS [not used]
def derivative_tuckerdecomp_wrt_weight_core_matrix(data, class_12345or6, labels_onehotencoded, tuckerdecomp_rank_1, tuckerdecomp_rank_2, tuckerdecomp_rank_3, bias_vector, factormatrix1_class_12345or6, factormatrix2_class_12345or6, factormatrix3_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    derivative_log_likelihood_weight_coretensor_list = []
    
    for i in range(len(labels_onehotencoded)):
        # define an empty matrix which will contain all values of term 1 for the partial derivative of log-likelihood wrt core matrix of ANY CLASS r, i.e., \frac{dl}{dH^{(r)}} for r = 1, ..., 6
        # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt core matrix of ANY CLASS r
        #term_1_weight_coretensor = np.zeros(shape = (tuckerdecomp_rank_3, tuckerdecomp_rank_1, tuckerdecomp_rank_2)) UNNECESSARY
        
        # this list is just to contain all values for a specific partial derivative of log-likelihood wrt h^{(r)}_{tuckerdecomp_rank_3, tuckerdecomp_rank_1, tuckerdecomp_rank_2} where this will be done for each tuckerdecomp_rank_1 = 1, ..., P_1, tuckerdecomp_rank_2 = 1, ..., P_2 and tuckerdecomp_rank_3 = 1, ..., P_3
        term_1_needtosum_list = []
        
        # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt core matrix of ANY CLASS r, i.e., \frac{dl}{dH^{(r)}}
        derivative_log_likelihood_weight_coretensor_ithdata = np.zeros(shape = (tuckerdecomp_rank_3, tuckerdecomp_rank_1, tuckerdecomp_rank_2))
        
        for j1 in range(tuckerdecomp_rank_1):
            for j2 in range(tuckerdecomp_rank_2):
                for j3 in range(tuckerdecomp_rank_3):
                    for x in range(240):
                        for y in range(320):
                            for z in range(3):
                                # We differentiated  
                                term_1_needtosum = data[i][z, x, y] * factormatrix1_class_12345or6[x, j1] * factormatrix2_class_12345or6[y, j2] * factormatrix3_class_12345or6[z, j3]
                    
                                term_1_needtosum_list.append(term_1_needtosum)
                
                    term_1 = sum(term_1_needtosum_list)
                    
                    #term_1_weight_coretensor[j3, j1, j2] = term_1
                    
                    # refresh this back to empty
                    term_1_needtosum_list = []
                    
                    # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt core matrix of ANY CLASS r, i.e., \frac{dl}{dH^{(r)}}
                    # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt core matrix of ANY CLASS
                    #term_2_weight_coretensor = np.zeros(shape = (tuckerdecomp_rank_3, tuckerdecomp_rank_1, tuckerdecomp_rank_2)) UNNECESSARY
                    
                    
                    if class_12345or6 == 1:
                        term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_coretensor[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(term_1 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                    
                    elif class_12345or6 == 2:
                        term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_coretensor[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(term_1 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                    
                    elif class_12345or6 == 3:
                        term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_coretensor[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(term_1 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                
                    elif class_12345or6 == 4:
                        term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_coretensor[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(term_1 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                    
                    elif class_12345or6 == 5:
                        term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_corematrix[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(term_1 - term_2) + labels_onehotencoded[i][5]*(0 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                    
                    else:
                        term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list[i])) / np.exp(ln_of_innerproduct_exponentials_list[i])
                        
                        #term_2_weight_coretensor[j3, j1, j2] = term_2
                        
                        # Calculate the partial derivative of log-likelihood wrt weight value
                        derivative_log_likelihood = labels_onehotencoded[i][0]*(0 - term_2) + labels_onehotencoded[i][1]*(0 - term_2) + labels_onehotencoded[i][2]*(0 - term_2) + labels_onehotencoded[i][3]*(0 - term_2) + labels_onehotencoded[i][4]*(0 - term_2) + labels_onehotencoded[i][5]*(term_1 - term_2) + labels_onehotencoded[i][6]*(0 - term_2)
                        
                        derivative_log_likelihood_weight_coretensor_ithdata[j3, j1, j2] = derivative_log_likelihood
                
        derivative_log_likelihood_weight_coretensor_list.append(derivative_log_likelihood_weight_coretensor_ithdata)
            
    
    derivative_log_likelihood_weight_coretensor = sum(derivative_log_likelihood_weight_coretensor_list)
    
    return derivative_log_likelihood_weight_coretensor

# To be used prior to predicting the labels [NOT USED ANYMORE] 
# TRIED TO USE IT FOR FINDING THE LOG-LIKELIHOOD PER MINI-BATCH BUT ENDED UP BEING SLOWER THAN THE MANUAL APPROACH
#def outer_product_factor_matrices_perclass(cpdecomp_rank, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
#                           , factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6):
#    
#    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
#    outer_product_factor_matrices_class1_list = []
#    outer_product_factor_matrices_class2_list = []
#    outer_product_factor_matrices_class3_list = []
#    outer_product_factor_matrices_class4_list = []
#    outer_product_factor_matrices_class5_list = []
#    outer_product_factor_matrices_class6_list = []
#    
#    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
#    for j in range(cpdecomp_rank):
#        outer_product_factor_matrices_class1 = outer2(factor_matrix3_class1[:,j], factor_matrix1_class1[:,j], factor_matrix2_class1[:,j])
#        outer_product_factor_matrices_class2 = outer2(factor_matrix3_class2[:,j], factor_matrix1_class2[:,j], factor_matrix2_class2[:,j])
#        outer_product_factor_matrices_class3 = outer2(factor_matrix3_class3[:,j], factor_matrix1_class3[:,j], factor_matrix2_class3[:,j])
#        outer_product_factor_matrices_class4 = outer2(factor_matrix3_class4[:,j], factor_matrix1_class4[:,j], factor_matrix2_class4[:,j])
#        outer_product_factor_matrices_class5 = outer2(factor_matrix3_class5[:,j], factor_matrix1_class5[:,j], factor_matrix2_class5[:,j])
#        outer_product_factor_matrices_class6 = outer2(factor_matrix3_class6[:,j], factor_matrix1_class6[:,j], factor_matrix2_class6[:,j])
#        
#        outer_product_factor_matrices_class1_list.append(outer_product_factor_matrices_class1)
#        outer_product_factor_matrices_class2_list.append(outer_product_factor_matrices_class2)
#        outer_product_factor_matrices_class3_list.append(outer_product_factor_matrices_class3)
#        outer_product_factor_matrices_class4_list.append(outer_product_factor_matrices_class4)
#        outer_product_factor_matrices_class5_list.append(outer_product_factor_matrices_class5)
#        outer_product_factor_matrices_class6_list.append(outer_product_factor_matrices_class6)
#    
#    return outer_product_factor_matrices_class1_list, outer_product_factor_matrices_class2_list, outer_product_factor_matrices_class3_list, outer_product_factor_matrices_class4_list, outer_product_factor_matrices_class5_list, outer_product_factor_matrices_class6_list

#  [NOT USED ANYMORE] 
def train_predictedlabels(data, bias_vector, outer_product_factor_matrices_class1_list, outer_product_factor_matrices_class2_list, outer_product_factor_matrices_class3_list, outer_product_factor_matrices_class4_list, outer_product_factor_matrices_class5_list, outer_product_factor_matrices_class6_list):
    ## Assuming that we have finished training, the factor matrices found can now be used for prediction
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image train data
    # For the training set:
    train_set_predicted_labels = []
    train_prob_allclasses_perdata = np.zeros(shape = (len(data), 7)) # hosts the estimated probabilities for each observation
    
    for i in range(len(data)):
        # ^ getting sample size from that
        
        # Sum all them up in the list and then take the inner product of them with the ith data
        inner_product_class1 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class1_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class2 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class2_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class3 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class3_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class4 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class4_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class5 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class5_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class6 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class6_list, axis = 0)), vectorize_tensor_3D(data[i]))
        
        # Calculate the other logarithmic term needed in the log-likelihood (CAN'T BE USED SINCE IT LEAD TO OVERFLOWS IN EXPONENTIALS)
        #ln_of_innerproduct_exponentials = np.log(1 + np.exp(bias_vector[0] + inner_product_class1) + np.exp(bias_vector[1] + inner_product_class2) + np.exp(bias_vector[2] + inner_product_class3) + np.exp(bias_vector[3] + inner_product_class4) + np.exp(bias_vector[4] + inner_product_class5) + np.exp(bias_vector[5] + inner_product_class6))
        
        # TO CALCULATE ln_of_innerproduct_exponentials I'LL BE DOING A ROUNDABOUT METHOD. TESTED IT FOR SMALL VALUES DEFINED IN AND WORKED CORRECTLY. NOT SURE FOR LARGER VALUES IF FINE BUT GOOD ENOUGH
        # GOT IDEA FROM https://stackoverflow.com/questions/44033533/how-to-deal-with-exponent-overflow-of-64float-precision-in-python , https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html AND https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp
        
        ln_of_innerproduct_exponentials = logsumexp([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]])
        
        # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
        #temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
        #ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
        #ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]
        
        # Predict probabilities for each class
        prob_class1 = np.exp(bias_vector[0] + inner_product_class1) / np.exp(ln_of_innerproduct_exponentials)
        prob_class2 = np.exp(bias_vector[1] + inner_product_class2) / np.exp(ln_of_innerproduct_exponentials)
        prob_class3 = np.exp(bias_vector[2] + inner_product_class3) / np.exp(ln_of_innerproduct_exponentials)
        prob_class4 = np.exp(bias_vector[3] + inner_product_class4) / np.exp(ln_of_innerproduct_exponentials)
        prob_class5 = np.exp(bias_vector[4] + inner_product_class5) / np.exp(ln_of_innerproduct_exponentials)
        prob_class6 = np.exp(bias_vector[5] + inner_product_class6) / np.exp(ln_of_innerproduct_exponentials)
        
        # In case of nans (happens when inner_product_class is inf and ln_of_innerproduct_exponentials is inf), to return 0
        if math.isnan(prob_class1):
            prob_class1 = 0
        else:
            pass
        
        if math.isnan(prob_class2):
            prob_class2 = 0
        else:
            pass
        
        if math.isnan(prob_class3):
            prob_class3 = 0
        else:
            pass
        
        if math.isnan(prob_class4):
            prob_class4 = 0
        else:
            pass
        
        if math.isnan(prob_class5):
            prob_class5 = 0
        else:
            pass
        
        if math.isnan(prob_class6):
            prob_class6 = 0
        else:
            pass
        
        prob_class7 = 1 - prob_class1 - prob_class2 - prob_class3 - prob_class4 - prob_class5 - prob_class6
        
        prob_allclasses = [prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7]
        prob_allclasses = np.array(prob_allclasses)
        prob_allclasses = prob_allclasses.reshape(-1,1).T
        
        train_prob_allclasses_perdata[i] = prob_allclasses
        
        # Depending on which probability is higher, the ith data is classified to it
        if max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class1:
            train_set_predicted_labels.append(0)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class2:
            train_set_predicted_labels.append(1)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class3:
            train_set_predicted_labels.append(2)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class4:
            train_set_predicted_labels.append(3)
        
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class5:
            train_set_predicted_labels.append(4)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class6:
            train_set_predicted_labels.append(5)
        
        else:
            train_set_predicted_labels.append(6)

    return train_set_predicted_labels, train_prob_allclasses_perdata

# [NOT USED ANYMORE] 
def test_predictedlabels(data, bias_vector, outer_product_factor_matrices_class1_list, outer_product_factor_matrices_class2_list, outer_product_factor_matrices_class3_list, outer_product_factor_matrices_class4_list, outer_product_factor_matrices_class5_list, outer_product_factor_matrices_class6_list):
    ## Assuming that we have finished training, the factor matrices found can now be used for prediction
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image train data
    # For the training set:
    test_set_predicted_labels = []
    test_prob_allclasses_perdata = np.zeros(shape = (len(data), 7)) # hosts the estimated probabilities for each observation
    
    for i in range(len(data)):
        # ^ getting sample size from that
        
        # Sum all them up in the list and then take the inner product of them with the ith data
        inner_product_class1 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class1_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class2 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class2_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class3 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class3_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class4 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class4_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class5 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class5_list, axis = 0)), vectorize_tensor_3D(data[i]))
        inner_product_class6 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class6_list, axis = 0)), vectorize_tensor_3D(data[i]))
        
        # Calculate the other logarithmic term needed in the log-likelihood (CAN'T BE USED SINCE IT LEAD TO OVERFLOWS IN EXPONENTIALS)
        #ln_of_innerproduct_exponentials = np.log(1 + np.exp(bias_vector[0] + inner_product_class1) + np.exp(bias_vector[1] + inner_product_class2) + np.exp(bias_vector[2] + inner_product_class3) + np.exp(bias_vector[3] + inner_product_class4) + np.exp(bias_vector[4] + inner_product_class5) + np.exp(bias_vector[5] + inner_product_class6))
        
        # TO CALCULATE ln_of_innerproduct_exponentials I'LL BE DOING A ROUNDABOUT METHOD. TESTED IT FOR SMALL VALUES DEFINED IN AND WORKED CORRECTLY. NOT SURE FOR LARGER VALUES IF FINE BUT GOOD ENOUGH
        # GOT IDEA FROM https://stackoverflow.com/questions/44033533/how-to-deal-with-exponent-overflow-of-64float-precision-in-python , https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html AND https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp
        
        ln_of_innerproduct_exponentials = logsumexp([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]])
        
        # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
        #temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
        #ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
        #ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]
        
        # Predict probabilities for each class
        prob_class1 = np.exp(bias_vector[0] + inner_product_class1) / np.exp(ln_of_innerproduct_exponentials)
        prob_class2 = np.exp(bias_vector[1] + inner_product_class2) / np.exp(ln_of_innerproduct_exponentials)
        prob_class3 = np.exp(bias_vector[2] + inner_product_class3) / np.exp(ln_of_innerproduct_exponentials)
        prob_class4 = np.exp(bias_vector[3] + inner_product_class4) / np.exp(ln_of_innerproduct_exponentials)
        prob_class5 = np.exp(bias_vector[4] + inner_product_class5) / np.exp(ln_of_innerproduct_exponentials)
        prob_class6 = np.exp(bias_vector[5] + inner_product_class6) / np.exp(ln_of_innerproduct_exponentials)
        
        # In case of nans (happens when inner_product_class is inf and ln_of_innerproduct_exponentials is inf), to return 0
        if math.isnan(prob_class1):
            prob_class1 = 0
        else:
            pass
        
        if math.isnan(prob_class2):
            prob_class2 = 0
        else:
            pass
        
        if math.isnan(prob_class3):
            prob_class3 = 0
        else:
            pass
        
        if math.isnan(prob_class4):
            prob_class4 = 0
        else:
            pass
        
        if math.isnan(prob_class5):
            prob_class5 = 0
        else:
            pass
        
        if math.isnan(prob_class6):
            prob_class6 = 0
        else:
            pass
        
        prob_class7 = 1 - prob_class1 - prob_class2 - prob_class3 - prob_class4 - prob_class5 - prob_class6
        
        prob_allclasses = [prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7]
        prob_allclasses = np.array(prob_allclasses)
        prob_allclasses = prob_allclasses.reshape(-1,1).T
        
        test_prob_allclasses_perdata[i] = prob_allclasses
        
        # Depending on which probability is higher, the ith data is classified to it
        if max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class1:
            test_set_predicted_labels.append(0)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class2:
            test_set_predicted_labels.append(1)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class3:
            test_set_predicted_labels.append(2)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class4:
            test_set_predicted_labels.append(3)
        
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class5:
            test_set_predicted_labels.append(4)
            
        elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class6:
            test_set_predicted_labels.append(5)
        
        else:
            test_set_predicted_labels.append(6)

    return test_set_predicted_labels, test_prob_allclasses_perdata




########################################################################################################
### Block relaxation algorithm using the training set - Gradient ascent (GA) optimizer with momentum ###
## CP DECOMPOSITION
# All blocks that we want to maximize by
Initial_blocks = [factormatrix_cpdecomp_1_CLASS1, factormatrix_cpdecomp_2_CLASS1, factormatrix_cpdecomp_3_CLASS1, factormatrix_cpdecomp_1_CLASS2, factormatrix_cpdecomp_2_CLASS2, factormatrix_cpdecomp_3_CLASS2, factormatrix_cpdecomp_1_CLASS3, factormatrix_cpdecomp_2_CLASS3, factormatrix_cpdecomp_3_CLASS3, factormatrix_cpdecomp_1_CLASS4, factormatrix_cpdecomp_2_CLASS4, factormatrix_cpdecomp_3_CLASS4, factormatrix_cpdecomp_1_CLASS5, factormatrix_cpdecomp_2_CLASS5, factormatrix_cpdecomp_3_CLASS5, factormatrix_cpdecomp_1_CLASS6, factormatrix_cpdecomp_2_CLASS6, factormatrix_cpdecomp_3_CLASS6]

learning_rate = loguniform.rvs(0.000001, 1, size=1).item() #tested 0.1, 0.01 and 0.001 so far and all diverged in 1-2 epochs. 0.00001 never moved after 18 epochs of training and only classified label 2
momentum_term = random.uniform(0.1, 0.999)
maxiter = 60

# tolerance parameter
#tol = 0.0000001

# lasso regularization parameter
lasso_regparameter = loguniform.rvs(0.000001, 0.1, size=1).item()

# Taking the initialized bias values we defined earlier, we are going to use gradient ascent to modify their initialized values with all factor matrices taken as 0 matrices.
#zeromatrix_1 = np.zeros(shape = (240,cpdecomp_rank))
#zeromatrix_2 = np.zeros(shape = (320,cpdecomp_rank))
#zeromatrix_3 = np.zeros(shape = (3,cpdecomp_rank))

# List of hyperparameters
hyperparameters = [batch_size, cpdecomp_rank, learning_rate, momentum_term, lasso_regparameter, maxiter]

# Time it
start = time.perf_counter()

#outer_product_factor_matrices_all_ranksandclasses = np.zeros(shape = (240,320,3))

inner_product_class1 = 0
inner_product_class2 = 0
inner_product_class3 = 0
inner_product_class4 = 0
inner_product_class5 = 0
inner_product_class6 = 0

# using the zero factor matrices, now to find the initial bias
for t3 in range(maxiter):
    # Shuffling the training data and labels in between epochs WHILE making sure that the shuffling isnt confusing anything
    random.Random(4).shuffle(data_tensors_train_random_batches)
    random.Random(4).shuffle(img_labels_train_onehotencoded_random_batches)
    # The sorting above was checked using the following example:
    # x = [[i] for i in range(10)] and y = [[0],[1],[0],[0],[1],[0],[0],[0],[0],[0]] and then doing random.Random(4).shuffle(x) and random.Random(4).shuffle(y). The sorted labels are matching the respective data
    
    # List containing all bias log-likelihoods
    bias_log_likelihood_total_list = []
    
    # Reset the grace
    grace_bias_initial = 0
    
    for minibatch in range(math.ceil(len(data_tensors_train) / batch_size)):
        # Calculate the log-likelihood and inner products etc
        inner_product_class1_list_new = []
        inner_product_class2_list_new = []
        inner_product_class3_list_new = []
        inner_product_class4_list_new = []
        inner_product_class5_list_new = []
        inner_product_class6_list_new = []
        ln_of_innerproduct_exponentials_list_new = []
        
        # inner products are all 0s anyway due to factor matrices of zeros
        #outer_product_factor_matrices_class1_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])
        #outer_product_factor_matrices_class2_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])
        #outer_product_factor_matrices_class3_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])
        #outer_product_factor_matrices_class4_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])
        #outer_product_factor_matrices_class5_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])
        #outer_product_factor_matrices_class6_rank1 = outer2(zeromatrix_1[:,0], zeromatrix_2[:,0], zeromatrix_3[:,0])

        #outer_product_factor_matrices_class1_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])
        #outer_product_factor_matrices_class2_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])
        #outer_product_factor_matrices_class3_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])
        #outer_product_factor_matrices_class4_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])
        #outer_product_factor_matrices_class5_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])
        #outer_product_factor_matrices_class6_rank2 = outer2(zeromatrix_1[:,1], zeromatrix_2[:,1], zeromatrix_3[:,1])

        #outer_product_factor_matrices_class1_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])
        #outer_product_factor_matrices_class2_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])
        #outer_product_factor_matrices_class3_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])
        #outer_product_factor_matrices_class4_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])
        #outer_product_factor_matrices_class5_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])
        #outer_product_factor_matrices_class6_rank3 = outer2(zeromatrix_1[:,2], zeromatrix_2[:,2], zeromatrix_3[:,2])

        for i in range(len(data_tensors_train_random_batches[minibatch])):
            # Calculate the other logarithmic term needed in the log-likelihood (CAN'T BE USED SINCE IT LEAD TO OVERFLOWS IN EXPONENTIALS)
            ln_of_innerproduct_exponentials = np.log(1 + np.exp(bias_cpdecomp_LIST[0] + inner_product_class1) + np.exp(bias_cpdecomp_LIST[1] + inner_product_class2) + np.exp(bias_cpdecomp_LIST[2] + inner_product_class3) + np.exp(bias_cpdecomp_LIST[3] + inner_product_class4) + np.exp(bias_cpdecomp_LIST[4] + inner_product_class5) + np.exp(bias_cpdecomp_LIST[5] + inner_product_class6))

            # TO CALCULATE ln_of_innerproduct_exponentials I'LL BE DOING A ROUNDABOUT METHOD. TESTED IT FOR SMALL VALUES DEFINED IN AND WORKED CORRECTLY. NOT SURE FOR LARGER VALUES IF FINE BUT GOOD ENOUGH
            # GOT IDEA FROM https://stackoverflow.com/questions/44033533/how-to-deal-with-exponent-overflow-of-64float-precision-in-python , https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html AND https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch.logsumexp
            
            ln_of_innerproduct_exponentials = logsumexp([[0], [bias_cpdecomp_LIST[0] + inner_product_class1], [bias_cpdecomp_LIST[1] + inner_product_class2], [bias_cpdecomp_LIST[2] + inner_product_class3], [bias_cpdecomp_LIST[3] + inner_product_class4], [bias_cpdecomp_LIST[4] + inner_product_class5], [bias_cpdecomp_LIST[5] + inner_product_class6]])
            
            # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
            #temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
            #ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
            #ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]

            # log-likelihood per sample i
            log_likelihood = img_labels_train_onehotencoded_random_batches[minibatch][i][0]*(bias_cpdecomp_LIST[0] + inner_product_class1 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][1]*(bias_cpdecomp_LIST[1] + inner_product_class2 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][2]*(bias_cpdecomp_LIST[2] + inner_product_class3 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][3]*(bias_cpdecomp_LIST[3] + inner_product_class4 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][4]*(bias_cpdecomp_LIST[4] + inner_product_class5 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][5]*(bias_cpdecomp_LIST[5] + inner_product_class6 - ln_of_innerproduct_exponentials) + img_labels_train_onehotencoded_random_batches[minibatch][i][6]*(0 - ln_of_innerproduct_exponentials)
            
            inner_product_class1_list_new.append(inner_product_class1)
            inner_product_class2_list_new.append(inner_product_class2)
            inner_product_class3_list_new.append(inner_product_class3)
            inner_product_class4_list_new.append(inner_product_class4)
            inner_product_class5_list_new.append(inner_product_class5)
            inner_product_class6_list_new.append(inner_product_class6)
            ln_of_innerproduct_exponentials_list_new.append(ln_of_innerproduct_exponentials)
        
        # Save parameters of best performing iteration
        #if bias_log_likelihood_total  == max(bias_log_likelihood_total_list):
        #    best_bias_cpdecomp_LIST = bias_cpdecomp_LIST
        
        #if len(bias_log_likelihood_total_list) > 1 and bias_log_likelihood_total_list[len(bias_log_likelihood_total_list) - 1] - bias_log_likelihood_total_list[len(bias_log_likelihood_total_list) - 2] < tol:
        #    grace_bias += 1
        #    
        #    if grace_bias == 2:
        #        break
        #    else:
        #        pass
        #else:
        #    grace_bias = 0
        derivative_log_likelihood_bias_vector = derivative_cportuckerdecomp_wrt_bias_vector(data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], bias_cpdecomp_LIST, inner_product_class1_list_new, inner_product_class2_list_new, inner_product_class3_list_new, inner_product_class4_list_new, inner_product_class5_list_new, inner_product_class6_list_new, ln_of_innerproduct_exponentials_list_new)
        
        if minibatch == 0:
            # Weight update rule GA + momentum - Initial iteration
            # Momentum is similar to the one used in Pytorch SGD
            first_momentum = derivative_log_likelihood_bias_vector
            
            updated_bias = bias_cpdecomp_LIST + learning_rate * first_momentum
            
            previous_momentum = first_momentum
            
            bias_cpdecomp_LIST = updated_bias
            
        else:
            # Weight update rule GA + momentum - All other iterations
            current_momentum = momentum_term * previous_momentum + derivative_log_likelihood_bias_vector
            
            updated_bias =  bias_cpdecomp_LIST + learning_rate * current_momentum
            
            previous_momentum = current_momentum
            
            bias_cpdecomp_LIST = updated_bias
            
    # Overwite the last iteration parameters with the best performing ones
    #bias_cpdecomp_LIST = best_bias_cpdecomp_LIST
    
finish = time.perf_counter()

print(f'Finished in {round(finish - start,2)} second(s)')

#best_bias_cpdecomp_LIST = np.array([ 1.010608  ,  1.41662179, -1.28453284, -1.5176887 , -1.27387486, -1.13802172])

# Lists containing all train and test overall log-likelihoods
train_overall_log_likelihood_GAwithmomentum_total_list = []
test_overall_log_likelihood_GAwithmomentum_total_list = []

# Lists containing all train and test performance metrics
train_accuracy_list = []
test_accuracy_list = []
train_confmatrix_list = []
test_confmatrix_list = []
train_f1_list = []
test_f1_list = []
train_auc_list = []
test_auc_list = []

# Grace period for overall Alternating Least Squares procedure
#grace_overall = 0

# Will be used in the algorithm. Since we have grace periods, this will be used to save the best performing parameters in each block update
#best_Initial_blocks = Initial_blocks

# Time it
start = time.perf_counter()

# Taking maxiter epochs
for t1 in range(maxiter):
    
    # Using the current iteration of parameters, calculate the log-likelihood
    with cf.ProcessPoolExecutor() as executor:
        train_overall_log_likelihood_list = list(executor.map(cpdecomp_loglikelihood_MultiProcess_py.cpdecomp_loglikelihood_MultiProcess_logl, data_tensors_train, img_labels_train_onehotencoded, itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17]), itertools.repeat(lasso_regparameter)))
    
    train_overall_log_likelihood_total = sum(train_overall_log_likelihood_list)
    
    train_overall_log_likelihood_GAwithmomentum_total_list.append(train_overall_log_likelihood_total)
    
    with cf.ProcessPoolExecutor() as executor:
        test_overall_log_likelihood_list = list(executor.map(cpdecomp_loglikelihood_MultiProcess_py.cpdecomp_loglikelihood_MultiProcess_logl, data_tensors_test, img_labels_test_onehotencoded, itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17]), itertools.repeat(lasso_regparameter))) # no regularization in the test
    
    test_overall_log_likelihood_total = sum(test_overall_log_likelihood_list)
    
    test_overall_log_likelihood_GAwithmomentum_total_list.append(test_overall_log_likelihood_total)
    
    print(f"Epoch {t1 + 1}, Train log-likelihood {train_overall_log_likelihood_total}, Test log-likelihood {test_overall_log_likelihood_total}")
    
    
    # Check if current overall log-likelihood is better than the previous
    #if t1 > 0 and train_overall_log_likelihood_GAwithmomentum_total_list[t1] - train_overall_log_likelihood_GAwithmomentum_total_list[t1-1] < tol:
    #    grace_overall += 1
    #    if grace_overall == 2:
    #        break
    #    else:
    #        pass
    #else:
    #    grace_overall = 0
    
    # Block updates
    for b in range(len(Initial_blocks)):
        
        print(f"Block {b + 1}")
        
        #for t2 in range(maxiter):
        # Shuffling the training data and labels in between epochs WHILE making sure that the shuffling isnt confusing anything
        random.Random(4).shuffle(data_tensors_train_random_batches)
        random.Random(4).shuffle(img_labels_train_onehotencoded_random_batches)
        
        # List containing all log-likelihoods for all iterations per factor matrix, which will then be reset when the next factor matrix is to be updated
        #block_log_likelihood_total_list = []
        
        # Reset the grace
        #grace_block = 0
        
        # UPDATE THESE IF CP DECOMP RANK CHANGES
        outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
        outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
        outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
        outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
        outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
        outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])

        outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
        outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
        outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
        outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
        outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
        outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])

        outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
        outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
        outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
        outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
        outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
        outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
        
        for minibatch in range(math.ceil(len(data_tensors_train) / batch_size)):
            ## Calculate the log-likelihood and inner products etc
            block_log_likelihood_list_total, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list = log_likelihood_innerproducts(data_tensors_train_random_batches, minibatch, outer_product_factor_matrices_class1_rank1, outer_product_factor_matrices_class1_rank2, outer_product_factor_matrices_class1_rank3, outer_product_factor_matrices_class2_rank1, outer_product_factor_matrices_class2_rank2, outer_product_factor_matrices_class2_rank3,
                    outer_product_factor_matrices_class3_rank1, outer_product_factor_matrices_class3_rank2, outer_product_factor_matrices_class3_rank3, outer_product_factor_matrices_class4_rank1, outer_product_factor_matrices_class4_rank2, outer_product_factor_matrices_class4_rank3,
                    outer_product_factor_matrices_class5_rank1, outer_product_factor_matrices_class5_rank2, outer_product_factor_matrices_class5_rank3, outer_product_factor_matrices_class6_rank1, outer_product_factor_matrices_class6_rank2, outer_product_factor_matrices_class6_rank3)
            
            
            # CALCULATING LASSO REGULARIZATION TERM
            sum_abs_factor_matrix_concatenated = np.sum(np.abs(np.concatenate((Initial_blocks[b][:,0], Initial_blocks[b][:,1], Initial_blocks[b][:,2]))))
            block_log_likelihood_list_total = block_log_likelihood_list_total - (lasso_regparameter * sum_abs_factor_matrix_concatenated)
            
            print(f"Epoch {t1 + 1}, Block Number {b + 1}, minibatch {minibatch + 1} of total minibatches {math.ceil(len(data_tensors_train) / batch_size)}, Block log-likelihood {block_log_likelihood_list_total}, Regularization term before tuning {sum_abs_factor_matrix_concatenated}")
            
            
            #block_log_likelihood_total_list.append(block_log_likelihood_total)
            
            # Check if current inner log-likelihood is better than the previous
            #if len(block_log_likelihood_total_list) > 1 and block_log_likelihood_total_list[len(block_log_likelihood_total_list) - 1] - block_log_likelihood_total_list[len(block_log_likelihood_total_list) - 2] < tol:
            #    grace_block += 1
            #    if grace_block == 2:
            #        break
            #    else:
            #        pass
            #else:
            #    grace_block = 0
            
            # IF condition to filter between the 18 different factor matrices which we will differentiate the log-likelihood with
            if b == 0:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                # Use multiprocessing to get the weight matrix derivative. Basically, this will get a list of weight matrix derivatives (up to n which is equal to the number of minibatches) which we will then sum
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS1_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class1, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), inner_product_class1_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS1 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS1_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS1
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    # Update the below with the updated block
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS1
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    # Update the below with the updated block
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
            elif b == 1:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS1_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class1, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), inner_product_class1_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS1 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS1_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS1
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS1
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
            elif b == 2:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS1_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class1, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), inner_product_class1_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS1 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS1_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS1
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS1
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
                    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
                    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
                    
            elif b == 3:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS2_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class2, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), inner_product_class2_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS2 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS2_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS2
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS2
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
            elif b == 4:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS2_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class2, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), inner_product_class2_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS2 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS2_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS2
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS2
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
            elif b == 5:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS2_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class2, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), inner_product_class2_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS2 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS2_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS2
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS2
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
                    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
                    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
                    
            elif b == 6:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS3_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class3, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), inner_product_class3_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS3 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS3_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS3
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS3
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
            elif b == 7:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS3_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class3, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), inner_product_class3_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS3 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS3_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS3
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS3
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
            elif b == 8:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS3_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class3, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), inner_product_class3_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS3 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS3_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS3
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS3
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
                    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
                    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
                    
            elif b == 9:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS4_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class4, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), inner_product_class4_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS4 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS4_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS4
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS4
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
            elif b == 10:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS4_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class4, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), inner_product_class4_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS4 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS4_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS4
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS4
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
            elif b == 11:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS4_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class4, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), inner_product_class4_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS4 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS4_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS4
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS4
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
                    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
                    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
                    
            elif b == 12:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS5_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class5, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), inner_product_class5_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS5 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS5_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS5
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS5
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
            elif b == 13:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS5_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class5, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), inner_product_class5_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS5 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS5_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS5
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS5
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
            elif b == 14:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS5_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class5, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), inner_product_class5_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS5 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS5_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS5
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS5
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
                    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
                    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
                    
            elif b == 15:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix1_CLASS6_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class6, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17]), inner_product_class6_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix1_CLASS6 = sum(derivative_log_likelihood_weight_factormatrix1_CLASS6_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix1_CLASS6
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix1_CLASS6
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
            elif b == 16:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix2_CLASS6_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class6, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17]), inner_product_class6_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix2_CLASS6 = sum(derivative_log_likelihood_weight_factormatrix2_CLASS6_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix2_CLASS6
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix2_CLASS6
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
            else:
                # Save parameters of best performing iteration
                #if block_log_likelihood_total  == max(block_log_likelihood_total_list):
                #    best_Initial_blocks[b] = Initial_blocks[b]
                
                with cf.ProcessPoolExecutor() as executor:
                    derivative_log_likelihood_weight_factormatrix3_CLASS6_list = list(executor.map(derivative_cpdecomp_wrt_weight_factor_matrices_MultiProcess_py.derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class6, data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17]), inner_product_class6_list, ln_of_innerproduct_exponentials_list, itertools.repeat(lasso_regparameter)))

                derivative_log_likelihood_weight_factormatrix3_CLASS6 = sum(derivative_log_likelihood_weight_factormatrix3_CLASS6_list)
                
                if minibatch == 0:
                    # Weight update rule GA + momentum - Initial iteration
                    first_momentum = derivative_log_likelihood_weight_factormatrix3_CLASS6
                    
                    updated_weight = Initial_blocks[b] + learning_rate * first_momentum
                    
                    previous_momentum = first_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
                else:
                    # Weight update rule GA + momentum - All other iterations
                    current_momentum =  momentum_term * previous_momentum + derivative_log_likelihood_weight_factormatrix3_CLASS6
                    
                    updated_weight =  Initial_blocks[b] + learning_rate * current_momentum
                    
                    previous_momentum = current_momentum
                    
                    Initial_blocks[b] = updated_weight
                    
                    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])
                    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])
                    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
                    
        
        # In between blocks b, after all the iterations t2 are done, it will set the current block b's parameters to the best one's that were spotted in that run
        #Initial_blocks[b] = best_Initial_blocks[b]
        
    # List containing all bias log-likelihoods per iteration
    #bias_log_likelihood_total_list = []
    
    # Reset the grace
    #grace_bias = 0
    
    outer_product_factor_matrices_class1_rank1 = outer2(Initial_blocks[0][:,0], Initial_blocks[1][:,0], Initial_blocks[2][:,0])
    outer_product_factor_matrices_class2_rank1 = outer2(Initial_blocks[3][:,0], Initial_blocks[4][:,0], Initial_blocks[5][:,0])
    outer_product_factor_matrices_class3_rank1 = outer2(Initial_blocks[6][:,0], Initial_blocks[7][:,0], Initial_blocks[8][:,0])
    outer_product_factor_matrices_class4_rank1 = outer2(Initial_blocks[9][:,0], Initial_blocks[10][:,0], Initial_blocks[11][:,0])
    outer_product_factor_matrices_class5_rank1 = outer2(Initial_blocks[12][:,0], Initial_blocks[13][:,0], Initial_blocks[14][:,0])
    outer_product_factor_matrices_class6_rank1 = outer2(Initial_blocks[15][:,0], Initial_blocks[16][:,0], Initial_blocks[17][:,0])

    outer_product_factor_matrices_class1_rank2 = outer2(Initial_blocks[0][:,1], Initial_blocks[1][:,1], Initial_blocks[2][:,1])
    outer_product_factor_matrices_class2_rank2 = outer2(Initial_blocks[3][:,1], Initial_blocks[4][:,1], Initial_blocks[5][:,1])
    outer_product_factor_matrices_class3_rank2 = outer2(Initial_blocks[6][:,1], Initial_blocks[7][:,1], Initial_blocks[8][:,1])
    outer_product_factor_matrices_class4_rank2 = outer2(Initial_blocks[9][:,1], Initial_blocks[10][:,1], Initial_blocks[11][:,1])
    outer_product_factor_matrices_class5_rank2 = outer2(Initial_blocks[12][:,1], Initial_blocks[13][:,1], Initial_blocks[14][:,1])
    outer_product_factor_matrices_class6_rank2 = outer2(Initial_blocks[15][:,1], Initial_blocks[16][:,1], Initial_blocks[17][:,1])

    outer_product_factor_matrices_class1_rank3 = outer2(Initial_blocks[0][:,2], Initial_blocks[1][:,2], Initial_blocks[2][:,2])
    outer_product_factor_matrices_class2_rank3 = outer2(Initial_blocks[3][:,2], Initial_blocks[4][:,2], Initial_blocks[5][:,2])
    outer_product_factor_matrices_class3_rank3 = outer2(Initial_blocks[6][:,2], Initial_blocks[7][:,2], Initial_blocks[8][:,2])
    outer_product_factor_matrices_class4_rank3 = outer2(Initial_blocks[9][:,2], Initial_blocks[10][:,2], Initial_blocks[11][:,2])
    outer_product_factor_matrices_class5_rank3 = outer2(Initial_blocks[12][:,2], Initial_blocks[13][:,2], Initial_blocks[14][:,2])
    outer_product_factor_matrices_class6_rank3 = outer2(Initial_blocks[15][:,2], Initial_blocks[16][:,2], Initial_blocks[17][:,2])
    
    for minibatch in range(math.ceil(len(data_tensors_train) / batch_size)):
        # Calculate the log-likelihood and inner products etc
        _, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list = log_likelihood_innerproducts(data_tensors_train_random_batches, minibatch, outer_product_factor_matrices_class1_rank1, outer_product_factor_matrices_class1_rank2, outer_product_factor_matrices_class1_rank3, outer_product_factor_matrices_class2_rank1, outer_product_factor_matrices_class2_rank2, outer_product_factor_matrices_class2_rank3,
                outer_product_factor_matrices_class3_rank1, outer_product_factor_matrices_class3_rank2, outer_product_factor_matrices_class3_rank3, outer_product_factor_matrices_class4_rank1, outer_product_factor_matrices_class4_rank2, outer_product_factor_matrices_class4_rank3,
                outer_product_factor_matrices_class5_rank1, outer_product_factor_matrices_class5_rank2, outer_product_factor_matrices_class5_rank3, outer_product_factor_matrices_class6_rank1, outer_product_factor_matrices_class6_rank2, outer_product_factor_matrices_class6_rank3)
        
        # Save parameters of best performing iteration
        #if bias_log_likelihood_total  == max(bias_log_likelihood_total_list):
        #    best_bias_cpdecomp_LIST = bias_cpdecomp_LIST
        
        #if len(bias_log_likelihood_total_list) > 1 and bias_log_likelihood_total_list[len(bias_log_likelihood_total_list) - 1] - bias_log_likelihood_total_list[len(bias_log_likelihood_total_list) - 2] < tol:
        #    grace_bias += 1
        #    
        #    if grace_bias == 2:
        #        break
        #    else:
        #        pass
        #else:
        #    grace_bias = 0
        derivative_log_likelihood_bias_vector = derivative_cportuckerdecomp_wrt_bias_vector(data_tensors_train_random_batches[minibatch], img_labels_train_onehotencoded_random_batches[minibatch], bias_cpdecomp_LIST, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list)
        
        if minibatch == 0:
            # Weight update rule GA + momentum - Initial iteration
            first_momentum = derivative_log_likelihood_bias_vector
            
            updated_bias = bias_cpdecomp_LIST + learning_rate * first_momentum
            
            previous_momentum = first_momentum
            
            bias_cpdecomp_LIST = updated_bias
            
        else:
            # Weight update rule GA + momentum - All other iterations
            current_momentum = momentum_term * previous_momentum + derivative_log_likelihood_bias_vector
            
            updated_bias = bias_cpdecomp_LIST + learning_rate * current_momentum
            
            previous_momentum = current_momentum
            
            bias_cpdecomp_LIST = updated_bias
    
    # Set bias_cpdecomp_LIST equal to the bias which gave the highest log-likelihood
    #bias_cpdecomp_LIST = best_bias_cpdecomp_LIST
    
    ## Assuming that we have finished training, the factor matrices found can now be used for prediction
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image train data
    # For the training set:
    train_prob_allclasses_perdata = np.zeros(shape = (len(data_tensors_train), 7)) # hosts the estimated probabilities for each observation
    
    with cf.ProcessPoolExecutor() as executor:
        twovariables_alltrain_list = list(executor.map(predicted_labels_MultiProcess_py.train_predictedlabels_MultiProcess, data_tensors_train, itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17])))
    
    train_set_predicted_labels = [item[0] for item in twovariables_alltrain_list]
    train_prob_allclasses_perdata = np.concatenate([item[1] for item in twovariables_alltrain_list], axis = 0) # [item[1] for item in twovariables_alltrain_list] returns list of arrays which np.concatenate combines to one array
    
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image test data
    # For the test set:
    test_prob_allclasses_perdata = np.zeros(shape = (len(data_tensors_test), 7)) # hosts the estimated probabilities for each observation
    
    with cf.ProcessPoolExecutor() as executor:
        twovariables_alltest_list = list(executor.map(predicted_labels_MultiProcess_py.test_predictedlabels_MultiProcess, data_tensors_test, itertools.repeat(cpdecomp_rank), itertools.repeat(bias_cpdecomp_LIST), itertools.repeat(Initial_blocks[0]), itertools.repeat(Initial_blocks[1]), itertools.repeat(Initial_blocks[2]), itertools.repeat(Initial_blocks[3]), itertools.repeat(Initial_blocks[4]), itertools.repeat(Initial_blocks[5]), itertools.repeat(Initial_blocks[6]), itertools.repeat(Initial_blocks[7]), itertools.repeat(Initial_blocks[8]), itertools.repeat(Initial_blocks[9]), itertools.repeat(Initial_blocks[10]), itertools.repeat(Initial_blocks[11]), itertools.repeat(Initial_blocks[12]), itertools.repeat(Initial_blocks[13]), itertools.repeat(Initial_blocks[14]), itertools.repeat(Initial_blocks[15]), itertools.repeat(Initial_blocks[16]), itertools.repeat(Initial_blocks[17])))

    test_set_predicted_labels = [item[0] for item in twovariables_alltest_list]
    test_prob_allclasses_perdata = np.concatenate([item[1] for item in twovariables_alltest_list], axis = 0) # [item[1] for item in twovariables_alltrain_list] returns list of arrays which np.concatenate combines to one array

    
    ### Calculating performance metrics
    ## Accuracy of train and test data
    correct_train = 0
    wrong_train = 0
    for i in range(len(true_labels_train)):
        if true_labels_train[i] == train_set_predicted_labels[i]:
            correct_train += 1
        else:
            wrong_train += 1

    Train_accuracy = (correct_train / len(true_labels_train)) * 100

    correct_test = 0
    wrong_test = 0
    for i in range(len(true_labels_test)):
        if true_labels_test[i] == test_set_predicted_labels[i]:
            correct_test += 1
        else:
            wrong_test += 1

    Test_accuracy = (correct_test / len(true_labels_test)) * 100

    ## Confusion matrix of train and test data
    #originals were a list of tensors, so this stacks them as one tensor (IS THIS NEEDED?)
    #training_set_predicted_labels = torch.stack(training_set_predicted_labels)
    #true_labels_train = torch.stack(true_labels_train)
        
    cf_matrix_train = confusion_matrix(true_labels_train, train_set_predicted_labels)

    #originals were a list of tensors, so this stacks them as one tensor (IS THIS NEEDED?)
    #test_set_predicted_labels = torch.stack(test_set_predicted_labels)
    #true_labels_test = torch.stack(true_labels_test)
        
    cf_matrix_test = confusion_matrix(true_labels_test, test_set_predicted_labels)

    ## F1
    precision_train = precision_score(true_labels_train, train_set_predicted_labels, average='macro', zero_division=0)
    recall_train = recall_score(true_labels_train, train_set_predicted_labels, average='macro', zero_division=0)
    f_measure_train = 2 * (precision_train * recall_train / (precision_train + recall_train))

    precision_test = precision_score(true_labels_test, test_set_predicted_labels, average='macro', zero_division=0)
    recall_test = recall_score(true_labels_test, test_set_predicted_labels, average='macro', zero_division=0)
    f_measure_test = 2 * (precision_test * recall_test / (precision_test + recall_test))

    ## Multiclass ROC/AUC using OvR approach
    # We will use the onehotencoded labels here, so using img_labels_train_onehotencoded and img_labels_test_onehotencoded
    # Transform predictted labels to numpy arrays and onehotencode them (UNNEEDED)
    #array_predicted_img_labels_train = np.array(train_set_predicted_labels)
    #array_predicted_img_labels_test = np.array(test_set_predicted_labels)

    #predicted_img_labels_train_onehotencoded = OneHotEncoder.fit_transform(array_predicted_img_labels_train)
    #predicted_img_labels_test_onehotencoded = OneHotEncoder.fit_transform(array_predicted_img_labels_test)
    
    # reshaping to use in roc_auc_score
    array_img_labels_train = array_img_labels_train.reshape(-1,)
    array_img_labels_test = array_img_labels_test.reshape(-1,)

    macro_roc_auc_ovo_train = roc_auc_score(array_img_labels_train, train_prob_allclasses_perdata, multi_class = "ovo", average = "macro")
    macro_roc_auc_ovo_test = roc_auc_score(array_img_labels_test, test_prob_allclasses_perdata, multi_class = "ovo", average = "macro")
    
    train_accuracy_list.append(Train_accuracy)
    test_accuracy_list.append(Test_accuracy)
    train_confmatrix_list.append(cf_matrix_train)
    test_confmatrix_list.append(cf_matrix_test)
    train_f1_list.append(f_measure_train)
    test_f1_list.append(f_measure_test)
    train_auc_list.append(macro_roc_auc_ovo_train)
    test_auc_list.append(macro_roc_auc_ovo_test)
    
    print('---------------------------------')
    print('Training accuracy for epoch %d: %d %%' % (t1 + 1, Train_accuracy))
    print('Training Precision %.3f, Recall %.3f, F-measure %.3f' % (precision_train, recall_train, f_measure_train))
    print('Training AUC %.3f' % (macro_roc_auc_ovo_train))
    print('-----------------------')
    print('Test accuracy for epoch %d: %d %%' % (t1 + 1, Test_accuracy))
    print('Test Precision %.3f, Recall %.3f, F-measure %.3f' % (precision_test, recall_test, f_measure_test))
    print('Test AUC %.3f' % (macro_roc_auc_ovo_test))
    print('-----------------------')
    print('Training Confusion Matrix')
    print(cf_matrix_train)
    print('-----------------------')
    print('Test Confusion Matrix')
    print(cf_matrix_test)
    print('-----------------------')
    print('Epoch: %d, Training Loss: %.3f' % (t1 + 1, train_overall_log_likelihood_total / len(data_tensors_train)))
    print('Epoch: %d, Test Loss: %.3f' % (t1 + 1, test_overall_log_likelihood_total / len(data_tensors_test)))
    print('---------------------------------')
    
    ## Save results
    
    with open('C:/Users/Andrew/Downloads/Results.txt', 'w') as file:
        # write variables using str() function
        file.write("Hyperparameters batch_size, cpdecomp_rank, learning_rate, momentum_term, lasso_regparameter, maxiter = " + str(hyperparameters) + '\n')
        file.write("Train Overall Log-likelihood per epoch = " + str(train_overall_log_likelihood_GAwithmomentum_total_list) + '\n')
        file.write("Test Overall Log-likelihood per epoch = " + str(test_overall_log_likelihood_GAwithmomentum_total_list) + '\n')
        file.write("Train Accuracy = " + str(train_accuracy_list) + '\n')
        file.write("Test Accuracy = " + str(test_accuracy_list) + '\n')
        file.write("Train F1 = " + str(train_f1_list) + '\n')
        file.write("Test F1 = " + str(test_f1_list) + '\n')
        file.write("Train AUC = " + str(train_auc_list) + '\n')
        file.write("Test AUC = " + str(test_auc_list) + '\n')
        file.write("Training Confusion matrix = " + str(train_confmatrix_list) + '\n')
        file.write("Test Confusion matrix = " + str(test_confmatrix_list) + '\n')

finish = time.perf_counter()

print(f'Finished in {round(finish - start,2)} second(s)')


# Save progress (used to in case of stopped runs or troubleshooting)
with open('C:/Users/Andrew/Downloads/SavedParameters.txt', 'w') as file:
    # write variables using str() function
    file.write("hyperparameters = " + str(hyperparameters) + '\n')
    file.write("Initial_blocks = " + str(Initial_blocks) + '\n')
    file.write("bias_cpdecomp_LIST = " + str(bias_cpdecomp_LIST) + '\n')
    file.write("train_overall_log_likelihood_GAwithmomentum_total_list = " + str(train_overall_log_likelihood_GAwithmomentum_total_list) + '\n')
    file.write("test_overall_log_likelihood_GAwithmomentum_total_list = " + str(test_overall_log_likelihood_GAwithmomentum_total_list) + '\n')
    file.write("train_accuracy_list = " + str(train_accuracy_list) + '\n')
    file.write("test_accuracy_list = " + str(test_accuracy_list) + '\n')
    file.write("train_f1_list = " + str(train_f1_list) + '\n')
    file.write("test_f1_list = " + str(test_f1_list) + '\n')
    file.write("train_auc_list = " + str(train_auc_list) + '\n')
    file.write("test_auc_list = " + str(test_auc_list) + '\n')
    file.write("train_confmatrix_list = " + str(train_confmatrix_list) + '\n')
    file.write("test_confmatrix_list = " + str(test_confmatrix_list) + '\n')

