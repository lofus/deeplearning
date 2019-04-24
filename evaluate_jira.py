import matplotlib.pyplot as plt
import numpy as np
from model import *
from load_data import *

train_x, train_y, test_x, test_y = load_data_jira()
print("#######################Load_data_jira########################")
print("train_x.shape ", train_x.shape)
print("train_y.shape ", train_y.shape)
print("test_x.shape ", test_x.shape)
print("test_y.shape ", test_y.shape)
print(train_x[:, :100])
print(train_y[:, :10])
print()
train_x = train_x / 1000
train_y = train_y / 20
test_x = test_x / 1000
test_y = test_y / 20
print(train_y[:, :10])


#Hyparameters
layers_dims = [train_x.shape[0], 64, 10, train_y.shape[0]]
learning_rate = 0.009
num_iterations = 6000
print_cost = True
print("#######################Hyparameters########################")
print("network layers_dims = ", layers_dims)
print("learning rate = ", learning_rate)
print("iterations = ", num_iterations)
print()

#Train model
print("#######################Training the model########################")
parameters, costs = L_layer_model(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost)
print("#######################Training completed!########################")
print()

#Evaluate
pred_train, train_accuracy = predict(train_x, train_y, parameters)
pred_test, test_accuracy = predict(test_x, test_y, parameters)
print("train_y   ", train_y[:, 500:525] * 10)
print("pred train", np.rint(pred_train[:, 500:525] * 10))
print("pred train", pred_train[:, 500:525] * 10)

print("test_y", test_y[:, 80:100] * 10)
print("pred_y", np.rint(pred_test[:, 80:100] * 10))
print("pred_y", pred_test[:, 80:100] * 10)

print("Train Accuracy = ", train_accuracy)
print("Test  Accuracy = ", test_accuracy)

#plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (hundred)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()
