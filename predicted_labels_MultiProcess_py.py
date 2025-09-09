import numpy as np
from functools import reduce
import warnings
import math
from scipy.special import logsumexp

## Ignore all errors (a bunch of overflow errors might come up from dividing by np.exp(ln_of_innerproduct_exponentials_list) but these will return 0 anyway)
warnings.filterwarnings("ignore", category = RuntimeWarning)

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
    # Define the channels as separate matrices
    tensorarray_1st = tensor[:,:,0]
    tensorarray_2nd = tensor[:,:,1]
    tensorarray_3rd = tensor[:,:,2]
    
    # Flatten them in column-major order
    tensorvector_1st = tensorarray_1st.flatten('F')
    tensorvector_2nd = tensorarray_2nd.flatten('F')
    tensorvector_3rd = tensorarray_3rd.flatten('F')
    
    # Stack them as one vector
    return np.hstack((tensorvector_1st, tensorvector_2nd, tensorvector_3rd))

# manually made, works good on an example tensor
def vectorize_tensor_3D_old(tensor):
    
    tensor_dimensions = tensor.shape # returns a tuple with the dimensions of the tensor
    # ^ THE FIRST DIMENSION IS THE NUMBER OF CHANNELS, THE SECOND IS THE NUMBER OF ROWS AND THE THIRD IS THE NUMBER OF COLUMNS
    vectorized_tensor = []
    
    for j in range(tensor_dimensions[0]):
        for k in range(tensor_dimensions[2]):
                
            vec = tensor[j,:,k]
                
            vectorized_tensor.extend(vec)

    return np.array(vectorized_tensor)

# To be used prior to predicting the labels (NOT USED)
def outer_product_factor_matrices_perclass(cpdecomp_rank, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
                           , factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6):
    
    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
    outer_product_factor_matrices_class1_list = []
    outer_product_factor_matrices_class2_list = []
    outer_product_factor_matrices_class3_list = []
    outer_product_factor_matrices_class4_list = []
    outer_product_factor_matrices_class5_list = []
    outer_product_factor_matrices_class6_list = []
    
    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
    for j in range(cpdecomp_rank):
        outer_product_factor_matrices_class1 = outer2(factor_matrix1_class1[:,j], factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
        outer_product_factor_matrices_class2 = outer2(factor_matrix1_class2[:,j], factor_matrix2_class2[:,j], factor_matrix3_class2[:,j])
        outer_product_factor_matrices_class3 = outer2(factor_matrix1_class3[:,j], factor_matrix2_class3[:,j], factor_matrix3_class3[:,j])
        outer_product_factor_matrices_class4 = outer2(factor_matrix1_class4[:,j], factor_matrix2_class4[:,j], factor_matrix3_class4[:,j])
        outer_product_factor_matrices_class5 = outer2(factor_matrix1_class5[:,j], factor_matrix2_class5[:,j], factor_matrix3_class5[:,j])
        outer_product_factor_matrices_class6 = outer2(factor_matrix1_class6[:,j], factor_matrix2_class6[:,j], factor_matrix3_class6[:,j])
        
        outer_product_factor_matrices_class1_list.append(outer_product_factor_matrices_class1)
        outer_product_factor_matrices_class2_list.append(outer_product_factor_matrices_class2)
        outer_product_factor_matrices_class3_list.append(outer_product_factor_matrices_class3)
        outer_product_factor_matrices_class4_list.append(outer_product_factor_matrices_class4)
        outer_product_factor_matrices_class5_list.append(outer_product_factor_matrices_class5)
        outer_product_factor_matrices_class6_list.append(outer_product_factor_matrices_class6)
    
    return outer_product_factor_matrices_class1_list, outer_product_factor_matrices_class2_list, outer_product_factor_matrices_class3_list, outer_product_factor_matrices_class4_list, outer_product_factor_matrices_class5_list, outer_product_factor_matrices_class6_list

def train_predictedlabels_MultiProcess(data, cpdecomp_rank, bias_vector, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
                           , factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6):
    ## Assuming that we have finished training, the factor matrices found can now be used for prediction
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image train data
    
    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
    outer_product_factor_matrices_class1_list = []
    outer_product_factor_matrices_class2_list = []
    outer_product_factor_matrices_class3_list = []
    outer_product_factor_matrices_class4_list = []
    outer_product_factor_matrices_class5_list = []
    outer_product_factor_matrices_class6_list = []

    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
    for j in range(cpdecomp_rank):
        outer_product_factor_matrices_class1 = outer2(factor_matrix1_class1[:,j], factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
        outer_product_factor_matrices_class2 = outer2(factor_matrix1_class2[:,j], factor_matrix2_class2[:,j], factor_matrix3_class2[:,j])
        outer_product_factor_matrices_class3 = outer2(factor_matrix1_class3[:,j], factor_matrix2_class3[:,j], factor_matrix3_class3[:,j])
        outer_product_factor_matrices_class4 = outer2(factor_matrix1_class4[:,j], factor_matrix2_class4[:,j], factor_matrix3_class4[:,j])
        outer_product_factor_matrices_class5 = outer2(factor_matrix1_class5[:,j], factor_matrix2_class5[:,j], factor_matrix3_class5[:,j])
        outer_product_factor_matrices_class6 = outer2(factor_matrix1_class6[:,j], factor_matrix2_class6[:,j], factor_matrix3_class6[:,j])
        
        outer_product_factor_matrices_class1_list.append(outer_product_factor_matrices_class1)
        outer_product_factor_matrices_class2_list.append(outer_product_factor_matrices_class2)
        outer_product_factor_matrices_class3_list.append(outer_product_factor_matrices_class3)
        outer_product_factor_matrices_class4_list.append(outer_product_factor_matrices_class4)
        outer_product_factor_matrices_class5_list.append(outer_product_factor_matrices_class5)
        outer_product_factor_matrices_class6_list.append(outer_product_factor_matrices_class6)
            
                
    # Sum all them up in the list and then take the inner product of them with the ith data
    inner_product_class1 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class1_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class2 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class2_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class3 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class3_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class4 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class4_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class5 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class5_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class6 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class6_list, axis = 0)), vectorize_tensor_3D(data))
    
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
    
    if math.isnan(prob_class2):
        prob_class2 = 0
    
    if math.isnan(prob_class3):
        prob_class3 = 0
    
    if math.isnan(prob_class4):
        prob_class4 = 0
    
    if math.isnan(prob_class5):
        prob_class5 = 0
    
    if math.isnan(prob_class6):
        prob_class6 = 0
    
    prob_class7 = 1 - prob_class1 - prob_class2 - prob_class3 - prob_class4 - prob_class5 - prob_class6
    
    prob_allclasses = [prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7]
    prob_allclasses = np.array(prob_allclasses)
    prob_allclasses = prob_allclasses.reshape(-1,1).T
    
    # Depending on which probability is higher, the ith data is classified to it
    if max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class1:
        train_set_predicted_label = 0
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class2:
        train_set_predicted_label = 1
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class3:
        train_set_predicted_label = 2
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class4:
        train_set_predicted_label = 3
    
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class5:
        train_set_predicted_label = 4
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class6:
        train_set_predicted_label = 5
    
    else:
        train_set_predicted_label = 6
        
    return train_set_predicted_label, prob_allclasses

def test_predictedlabels_MultiProcess(data, cpdecomp_rank, bias_vector, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
                           , factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6):
    ## Assuming that we have finished training, the factor matrices found can now be used for prediction
    # Using the estimated Initial_blocks to create the weight tensors and predict labels for the image train data
    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
    outer_product_factor_matrices_class1_list = []
    outer_product_factor_matrices_class2_list = []
    outer_product_factor_matrices_class3_list = []
    outer_product_factor_matrices_class4_list = []
    outer_product_factor_matrices_class5_list = []
    outer_product_factor_matrices_class6_list = []

    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
    for j in range(cpdecomp_rank):
        outer_product_factor_matrices_class1 = outer2(factor_matrix1_class1[:,j], factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
        outer_product_factor_matrices_class2 = outer2(factor_matrix1_class2[:,j], factor_matrix2_class2[:,j], factor_matrix3_class2[:,j])
        outer_product_factor_matrices_class3 = outer2(factor_matrix1_class3[:,j], factor_matrix2_class3[:,j], factor_matrix3_class3[:,j])
        outer_product_factor_matrices_class4 = outer2(factor_matrix1_class4[:,j], factor_matrix2_class4[:,j], factor_matrix3_class4[:,j])
        outer_product_factor_matrices_class5 = outer2(factor_matrix1_class5[:,j], factor_matrix2_class5[:,j], factor_matrix3_class5[:,j])
        outer_product_factor_matrices_class6 = outer2(factor_matrix1_class6[:,j], factor_matrix2_class6[:,j], factor_matrix3_class6[:,j])
        
        outer_product_factor_matrices_class1_list.append(outer_product_factor_matrices_class1)
        outer_product_factor_matrices_class2_list.append(outer_product_factor_matrices_class2)
        outer_product_factor_matrices_class3_list.append(outer_product_factor_matrices_class3)
        outer_product_factor_matrices_class4_list.append(outer_product_factor_matrices_class4)
        outer_product_factor_matrices_class5_list.append(outer_product_factor_matrices_class5)
        outer_product_factor_matrices_class6_list.append(outer_product_factor_matrices_class6)
        
    # Sum all them up in the list and then take the inner product of them with the ith data
    inner_product_class1 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class1_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class2 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class2_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class3 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class3_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class4 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class4_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class5 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class5_list, axis = 0)), vectorize_tensor_3D(data))
    inner_product_class6 = np.inner(vectorize_tensor_3D(np.sum(outer_product_factor_matrices_class6_list, axis = 0)), vectorize_tensor_3D(data))
    
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
    
    if math.isnan(prob_class2):
        prob_class2 = 0
    
    if math.isnan(prob_class3):
        prob_class3 = 0
    
    if math.isnan(prob_class4):
        prob_class4 = 0
    
    if math.isnan(prob_class5):
        prob_class5 = 0
    
    if math.isnan(prob_class6):
        prob_class6 = 0
    
    prob_class7 = 1 - prob_class1 - prob_class2 - prob_class3 - prob_class4 - prob_class5 - prob_class6
    
    prob_allclasses = [prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7]
    prob_allclasses = np.array(prob_allclasses)
    prob_allclasses = prob_allclasses.reshape(-1,1).T
    
    # Depending on which probability is higher, the ith data is classified to it
    if max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class1:
        test_set_predicted_label = 0
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class2:
        test_set_predicted_label = 1
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class3:
        test_set_predicted_label = 2
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class4:
        test_set_predicted_label = 3
    
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class5:
        test_set_predicted_label = 4
        
    elif max(prob_class1, prob_class2, prob_class3, prob_class4, prob_class5, prob_class6, prob_class7) == prob_class6:
        test_set_predicted_label = 5
    
    else:
        test_set_predicted_label = 6
        
    return test_set_predicted_label, prob_allclasses