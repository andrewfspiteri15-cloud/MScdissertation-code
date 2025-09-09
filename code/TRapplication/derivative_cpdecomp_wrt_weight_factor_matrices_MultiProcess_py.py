import numpy as np
import warnings
import math
from functools import reduce

## Ignore all errors (a bunch of overflow errors might come up from dividing by np.exp(ln_of_innerproduct_exponentials_list) but these will return 0 anyway)
warnings.filterwarnings("ignore", category = RuntimeWarning)

def outer2(*vs):
    return reduce(np.multiply.outer, vs)


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

### UNUSED
# FACTOR MATRIX 1 OF ANY CLASS
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess(data, class_12345or6, labels_onehotencoded, cpdecomp_rank, bias_vector, factormatrix2_class_12345or6, factormatrix3_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{1, cpdecomp_rank, x} where this will be done for each cpdecomp_rank = 1, ..., Z and x = 1, ..., 240
    term_1_needtosum_list = []
    
    # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 1 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_1}
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        for x in range(240):
            for y in range(320):
                for z in range(3):
                    # We differentiated (changed from z, x, y to x, y, z)
                    term_1_needtosum = data[x, y, z] * factormatrix2_class_12345or6[y, j] * factormatrix3_class_12345or6[z, j]
                
                    term_1_needtosum_list.append(term_1_needtosum)
        
            term_1 = sum(term_1_needtosum_list)
            
            #term_1_weight_factormatrix1[x, j] = term_1
            
            # refresh this back to empty
            term_1_needtosum_list = []
            
            
            if class_12345or6 == 1:
                term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(term_1 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
                
                
            elif class_12345or6 == 2:
                term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(term_1 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
                
            elif class_12345or6 == 3:
                term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(term_1 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
                
            elif class_12345or6 == 4:
                term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(term_1 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
                
            elif class_12345or6 == 5:
                term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(term_1 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
                
            else:
                term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix1[x, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(term_1 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix1_ithdata[x, j] = derivative_log_likelihood
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF ANY CLASS
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess(data, class_12345or6, labels_onehotencoded, cpdecomp_rank, bias_vector, factormatrix1_class_12345or6, factormatrix3_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{2, cpdecomp_rank, y} where this will be done for each cpdecomp_rank = 1, ..., Z and y = 1, ..., 320
    term_1_needtosum_list = []
    
    # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 2 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_2}
    derivative_log_likelihood_weight_factormatrix2_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        for y in range(320):
            for x in range(240):
                for z in range(3):
                    # We differentiated  
                    term_1_needtosum = data[x, y, z] * factormatrix1_class_12345or6[x, j] * factormatrix3_class_12345or6[z, j]
                
                    term_1_needtosum_list.append(term_1_needtosum)
            
            term_1 = sum(term_1_needtosum_list)
            
            #term_1_weight_factormatrix2[y, j] = term_1
            
            # refresh this back to empty
            term_1_needtosum_list = []
            
            # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt factor matrix 2 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_2}
            # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt factor matrix 2 of ANY CLASS
            #term_2_weight_factormatrix2 = np.zeros(shape = (320, cpdecomp_rank)) UNNECESSARY
            
            
            if class_12345or6 == 1:
                term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(term_1 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
            
            elif class_12345or6 == 2:
                term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(term_1 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
            
            elif class_12345or6 == 3:
                term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(term_1 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
            
            elif class_12345or6 == 4:
                term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(term_1 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
            
            elif class_12345or6 == 5:
                term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(term_1 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
            
            else:
                term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix2[y, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(term_1 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix2_ithdata[y, j] = derivative_log_likelihood
        
    return derivative_log_likelihood_weight_factormatrix2_ithdata

# FACTOR MATRIX 3 OF ANY CLASS
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess(data, class_12345or6, labels_onehotencoded, cpdecomp_rank, bias_vector, factormatrix1_class_12345or6, factormatrix2_class_12345or6, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    # this list is just to contain all values for a specific partial derivative of log-likelihood wrt w^{(r)}_{3, cpdecomp_rank, z} where this will be done for each cpdecomp_rank = 1, ..., Z and z = 1, 2, 3
    term_1_needtosum_list = []
    
    # define an empty matrix which will contain all values of the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
    derivative_log_likelihood_weight_factormatrix3_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        for z in range(3):
            for x in range(240):
                for y in range(320):
                    # We differentiated  
                    term_1_needtosum = data[x, y, z] * factormatrix1_class_12345or6[x, j] * factormatrix2_class_12345or6[y, j]
                
                    term_1_needtosum_list.append(term_1_needtosum)
        
            term_1 = sum(term_1_needtosum_list)
            
            #term_1_weight_factormatrix3[z, j] = term_1
            
            # refresh this back to empty
            term_1_needtosum_list = []
            
            # define an empty matrix which will contain all values of term 2 for the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
            # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt factor matrix 3 of ANY CLASS
            #term_2_weight_factormatrix3 = np.zeros(shape = (3, cpdecomp_rank)) UNNECESSARY
        
        
            if class_12345or6 == 1:
                term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(term_1 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
            
            elif class_12345or6 == 2:
                term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(term_1 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
            
            elif class_12345or6 == 3:
                term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(term_1 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
            
            elif class_12345or6 == 4:
                term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(term_1 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
            
            elif class_12345or6 == 5:
                term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(term_1 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
            
            else:
                term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
                
                if math.isnan(term_2):
                    term_2 = 0
                
                #term_2_weight_factormatrix3[z, j] = term_2
                
                # Calculate the partial derivative of log-likelihood wrt weight value
                derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(term_1 - term_2) + labels_onehotencoded[6]*(0 - term_2)
                
                derivative_log_likelihood_weight_factormatrix3_ithdata[z, j] = derivative_log_likelihood
        
    return derivative_log_likelihood_weight_factormatrix3_ithdata


# BIAS OF EVERY CLASS (slower than just not using multiprocessing)
def derivative_cportuckerdecomp_wrt_bias_vector_MultiProcess(data, labels_onehotencoded, bias_vector, inner_product_class1_list, inner_product_class2_list, inner_product_class3_list, inner_product_class4_list, inner_product_class5_list, inner_product_class6_list, ln_of_innerproduct_exponentials_list):
    
    # Term 1 is the derivative of the 'inner product of the weight matrix of ANY CLASS r with the ith image in the data' wrt bias of ANY CLASS r. The bias vector contains all values of each class
    #term_1_bias_vector = np.ones(shape = 6) UNNECESSARY
    term_1 = 1
    
    # Term 2 is the derivative of the 'ln(1+exp(inner product of the weight matrix of class 1 with the ith image in the data) + exp(inner product of the weight matrix of class 2 with the ith image in the data) + ... exp(inner product of the weight matrix of class 6 with the ith image in the data)' wrt bias vector (containing bias values of all classes)
    #term_2_bias_vector = np.zeros(shape = 6) UNNECESSARY
        
    # define an empty vector which will contain all values of the partial derivative of log-likelihood wrt factor matrix 3 of ANY CLASS r, i.e., \frac{dl}{dW^{(r)}_3}
    derivative_log_likelihood_bias_vector_ithdata = np.zeros(shape = 6)
    
    for k in range(6):
        if k == 0:
            
            term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(term_1 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
        
        elif k == 1:
            term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(term_1 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
        
        elif k == 2:
            term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(term_1 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
        
        elif k == 3:
            term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(term_1 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
        
        elif k == 4:
            term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(term_1 - term_2) + labels_onehotencoded[5]*(0 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
    
        else:
            term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
            
            if math.isnan(term_2):
                term_2 = 0
            
            #term_2_bias_vector[k] = term_2
            
            # Calculate the partial derivative of log-likelihood wrt weight value
            derivative_log_likelihood = labels_onehotencoded[0]*(0 - term_2) + labels_onehotencoded[1]*(0 - term_2) + labels_onehotencoded[2]*(0 - term_2) + labels_onehotencoded[3]*(0 - term_2) + labels_onehotencoded[4]*(0 - term_2) + labels_onehotencoded[5]*(term_1 - term_2) + labels_onehotencoded[6]*(0 - term_2)
            
            derivative_log_likelihood_bias_vector_ithdata[k] = derivative_log_likelihood
    
    return derivative_log_likelihood_bias_vector_ithdata


# FACTOR MATRIX 1 OF CLASS 1
#def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class1(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list):
#    
#    outer_product_class1_forderivative = []
#    for j in range(cpdecomp_rank):
#        outer_product = outer2(factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
#        
#        outer_product_class1_forderivative.append(outer_product)
#        
#    
#    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
#    
#    for j in range(cpdecomp_rank):
#        for i in range(240):
#            term_1 = np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:,i,:].T.flatten('F'))
#            
#            term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
#        
#            if math.isnan(term_2):
#                term_2 = 0
#        
#            # Calculate the partial derivative of log-likelihood wrt weight value
#            derivative_log_likelihood_weight_factormatrix1_ithdata[i, j] = labels_onehotencoded[0]*term_1 - term_2
#        
#    return derivative_log_likelihood_weight_factormatrix1_ithdata

### THESE ARE USED (created for use in multiprocessing)
# Making the generator and avoiding the for loop (not that much different performance-wise but around a second faster)
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class1(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class1[:,j], factor_matrix3_class1[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator [changed from :, i, : to i, :, :] due to python update
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i,:,:].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class1[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class2(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class2[:,j], factor_matrix3_class2[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i, :, :].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class2[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class3(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class3[:,j], factor_matrix3_class3[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i, :, :].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class3[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class4(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class4[:,j], factor_matrix3_class4[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i, :, :].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class4[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class5(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class5[:,j], factor_matrix3_class5[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i, :, :].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class5[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class6(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix2_class6[:,j], factor_matrix3_class6[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (240, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[i, :, :].T.flatten('F')) for i in range(240))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix1_class6[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 1
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class1(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class1[:,j], factor_matrix3_class1[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class1[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class2(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class2[:,j], factor_matrix3_class2[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class2[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class3(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class3[:,j], factor_matrix3_class3[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class3[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class4(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class4[:,j], factor_matrix3_class4[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class4[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class5(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class5[:,j], factor_matrix3_class5[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class5[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class6(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class6[:,j], factor_matrix3_class6[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (320, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, i, :].T.flatten('F')) for i in range(320))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix2_class6[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata


# FACTOR MATRIX 3 OF CLASS 1
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class1(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class1[:,j], factor_matrix2_class1[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[0] + inner_product_class1_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class1[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class2(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class2[:,j], factor_matrix2_class2[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[1] + inner_product_class2_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class2[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class3(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class3[:,j], factor_matrix2_class3[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[2] + inner_product_class3_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class3[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class4(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class4[:,j], factor_matrix2_class4[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[3] + inner_product_class4_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class4[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class5(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class5[:,j], factor_matrix2_class5[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[4] + inner_product_class5_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class5[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class6(data, labels_onehotencoded, cpdecomp_rank, bias_vector, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
    outer_product_class1_forderivative = []
    for j in range(cpdecomp_rank):
        outer_product = outer2(factor_matrix1_class6[:,j], factor_matrix2_class6[:,j])
        
        outer_product_class1_forderivative.append(outer_product)
        
    
    derivative_log_likelihood_weight_factormatrix1_ithdata = np.zeros(shape = (3, cpdecomp_rank))
    
    for j in range(cpdecomp_rank):
        # Create generator
        iterable = (np.inner(outer_product_class1_forderivative[j].flatten('F'), data[:, :, i].flatten('F')) for i in range(3))
        # Generator creates an array
        term_1 = np.fromiter(iterable, dtype = float)
            
        term_2 = (term_1 * np.exp(bias_vector[5] + inner_product_class6_list)) / np.exp(ln_of_innerproduct_exponentials_list)
        
        term_2[np.isnan(term_2)] = 0
        
        # Calculate the partial derivative of log-likelihood wrt weight value AND INCLUDING THE L1 REGULARIZATION WHICH IS "- lasso_regparameter * np.sign(factor_matrix3_class6[:,j])"
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata