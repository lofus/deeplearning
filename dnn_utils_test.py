from dnn_utils import *
from dnn_utils_test_cases import *


#Test linear_forward
print("################Test linear_forward##################")
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
print("linear_cache = " , linear_cache)
print()


#Test linear_activation_forward
print("################Test linear_activation_forward##################")
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))
print()

#Test initialize_parameters_deep
parameters = initialize_parameters_deep([5,4,3])
print("################Test initialize_parameters_deep##################")
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print()

#Test L_model_forward_test_case_2hidden
print("################Test L_model_forward_2hidden##################")
X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
print()

#Test for compute cost
print("################Test compute_cost##################")
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
print()


#Test linear_backward
print("################Test linear_backward##################")
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))
print()

#Test linear_activation_backward
print("################Test linear_activation_backward##################")
dAL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print("sigmoid:")
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db) + "\n")
dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print("relu:")
print("dA_prev = "+ str(dA_prev))
print("dW = " + str(dW))
print("db = " + str(db))
print()


#Test L_model_backward
print("################Test L_model_backward##################")
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)
print()

#Test update_parameters
print("################Test update_parameters##################")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print("W1 = "+ str(parameters["W1"]))
print("b1 = "+ str(parameters["b1"]))
print("W2 = "+ str(parameters["W2"]))
print("b2 = "+ str(parameters["b2"]))
print()


