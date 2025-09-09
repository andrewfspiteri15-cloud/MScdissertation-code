import numpy as np
import warnings
#import math
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

### THESE ARE USED (created for use in multiprocessing)
# Making the generator and avoiding the for loop (not that much different performance-wise but around a second faster)
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class1_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[0]*labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class2_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[1]*labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class3_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[2]*labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class4_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[3]*labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class5_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[4]*labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 1 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix1_MultiProcess_class6_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[5]*labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix1_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 1
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class1_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[0]*labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class2_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[1]*labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class3_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[2]*labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class4_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[3]*labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class5_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[4]*labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 2 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix2_MultiProcess_class6_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[5]*labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix2_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata


# FACTOR MATRIX 3 OF CLASS 1
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class1_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class1, factor_matrix2_class1, factor_matrix3_class1, inner_product_class1_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[0]*labels_onehotencoded[0]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class1[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 2
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class2_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class2, factor_matrix2_class2, factor_matrix3_class2, inner_product_class2_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[1]*labels_onehotencoded[1]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class2[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 3
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class3_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class3, factor_matrix2_class3, factor_matrix3_class3, inner_product_class3_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[2]*labels_onehotencoded[2]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class3[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 4
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class4_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class4, factor_matrix2_class4, factor_matrix3_class4, inner_product_class4_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[3]*labels_onehotencoded[3]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class4[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 5
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class5_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class5, factor_matrix2_class5, factor_matrix3_class5, inner_product_class5_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[4]*labels_onehotencoded[4]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class5[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata

# FACTOR MATRIX 3 OF CLASS 6
def derivative_cpdecomp_wrt_weight_factor_matrix3_MultiProcess_class6_weighted(data, labels_onehotencoded, cpdecomp_rank, bias_vector, class_weights, factor_matrix1_class6, factor_matrix2_class6, factor_matrix3_class6, inner_product_class6_list, ln_of_innerproduct_exponentials_list, lasso_regparameter):
    
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
        derivative_log_likelihood_weight_factormatrix1_ithdata[:, j] = class_weights[5]*labels_onehotencoded[5]*term_1 - term_2 - lasso_regparameter * np.sign(factor_matrix3_class6[:,j])
        
    return derivative_log_likelihood_weight_factormatrix1_ithdata