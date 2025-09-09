import numpy as np
from functools import reduce
import torch
#from scipy.special import logsumexp

def outer2(*vs):
    return reduce(np.multiply.outer, vs)
# ^ using the above since it worked with correctly with example u = [1, -2, 0] and v = [2, -2, 3, -6] in outer2(u,v)
# WORKED CORRECTLY WHEN I ADDED A CONSTANT WITH THE VECTORS, e.g. taking y = [6,7,9,2] and z = 5 in outer2(z,u,v,y)

# Note that I tried using np.einsum instead but above was faster

## VECTORIZE A 3D TENSOR

# manually made, compared with vectorize_tensor_3D_old, this is a numpy only alternative and comparing the results from both gives the same outputs
def vectorize_tensor_3D(tensor):
    # Define the channels as separate matrices
    # UPDATE THESE IF CP DECOMP RANK CHANGES
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

def outer_product_factor_matrices_classes(cpdecomp_rank, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
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

# HAS LASSO REGULARIZATION and has class_weights
def cpdecomp_loglikelihood_MultiProcess_logl_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3
                           , factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, lasso_regparameter):
    
    
    # Define empty lists to contain all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} so we can sum them later
    outer_product_factor_matrices_class1_list = []
    outer_product_factor_matrices_class2_list = []
    outer_product_factor_matrices_class3_list = []
    outer_product_factor_matrices_class4_list = []
    outer_product_factor_matrices_class5_list = []
    outer_product_factor_matrices_class6_list = []
    sum_abs_factor_matrix_concatenated = 0
    
    # Calculate all needed \sum_{z=1}^Z w^{(r)}_{1,z} o w^{(r)}_{2,z}, o w^{(r)}_{3,z} with for loop given cpdecomp_rank
    for j in range(cpdecomp_rank):
        outer_product_factor_matrices_class1 = outer2(factor_matrix1_class1[:,j], factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
        outer_product_factor_matrices_class2 = outer2(factor_matrix1_class2[:,j], factor_matrix2_class2[:,j], factor_matrix3_class2[:,j])
        outer_product_factor_matrices_class3 = outer2(factor_matrix1_class3[:,j], factor_matrix2_class3[:,j], factor_matrix3_class3[:,j])
        outer_product_factor_matrices_class4 = outer2(factor_matrix1_class4[:,j], factor_matrix2_class4[:,j], factor_matrix3_class4[:,j])
        outer_product_factor_matrices_class5 = outer2(factor_matrix1_class5[:,j], factor_matrix2_class5[:,j], factor_matrix3_class5[:,j])
        outer_product_factor_matrices_class6 = outer2(factor_matrix1_class6[:,j], factor_matrix2_class6[:,j], factor_matrix3_class6[:,j])
        
        # CALCULATING LASSO REGULARIZATION TERM
        sum_abs_factor_matrix_concatenated += np.sum(np.abs(np.concatenate((factor_matrix1_class1[:,j], factor_matrix1_class2[:,j], factor_matrix1_class3[:,j], factor_matrix1_class4[:,j], factor_matrix1_class5[:,j], factor_matrix1_class6[:,j], factor_matrix2_class1[:,j], factor_matrix2_class2[:,j], factor_matrix2_class3[:,j], factor_matrix2_class4[:,j], factor_matrix2_class5[:,j], factor_matrix2_class6[:,j], factor_matrix3_class1[:,j], factor_matrix3_class2[:,j], factor_matrix3_class3[:,j], factor_matrix3_class4[:,j], factor_matrix3_class5[:,j], factor_matrix3_class6[:,j]))))
        
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
    
    #ln_of_innerproduct_exponentials = logsumexp([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]])
    
    # torch version of the above (torch was not usable with numpy 2.0 so using above instead)
    temporary_tensor = torch.tensor([[0], [bias_vector[0] + inner_product_class1], [bias_vector[1] + inner_product_class2], [bias_vector[2] + inner_product_class3], [bias_vector[3] + inner_product_class4], [bias_vector[4] + inner_product_class5], [bias_vector[5] + inner_product_class6]], dtype = torch.float64)
    ln_of_innerproduct_exponentials = torch.logsumexp(temporary_tensor, 0)
    ln_of_innerproduct_exponentials = ln_of_innerproduct_exponentials.numpy()[0]
    
    # log-likelihood per sample
    log_likelihood = class_weights[0]*labels_onehotencoded[0]*(bias_vector[0] + inner_product_class1 - ln_of_innerproduct_exponentials) + class_weights[1]*labels_onehotencoded[1]*(bias_vector[1] + inner_product_class2 - ln_of_innerproduct_exponentials) + class_weights[2]*labels_onehotencoded[2]*(bias_vector[2] + inner_product_class3 - ln_of_innerproduct_exponentials) + class_weights[3]*labels_onehotencoded[3]*(bias_vector[3] + inner_product_class4 - ln_of_innerproduct_exponentials) + class_weights[4]*labels_onehotencoded[4]*(bias_vector[4] + inner_product_class5 - ln_of_innerproduct_exponentials) + class_weights[5]*labels_onehotencoded[5]*(bias_vector[5] + inner_product_class6 - ln_of_innerproduct_exponentials) + class_weights[6]*labels_onehotencoded[6]*(0 - ln_of_innerproduct_exponentials) - (lasso_regparameter * sum_abs_factor_matrix_concatenated)
    
    return log_likelihood
