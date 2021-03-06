import matplotlib.pyplot as plt
from model import *
from load_data import *

train_x, train_y, test_x, test_y, classes = load_data_cat_pics()
print("#######################Load_data_cat_pics########################")
print("train_x.shape ", train_x.shape)
print("train_y.shape ", train_y.shape)
print("test_x.shape ", test_x.shape)
print("test_y.shape ", test_y.shape)
print(train_x[:, :2])
print(train_y[:, :2])
print()

#Hyperparameters
layers_dims = [train_x.shape[0], 32, train_y.shape[0]]
learning_rate = 0.009
num_iterations = 1600
print_cost = True
print("#######################Hyperparameters########################")
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
pred_train, train_accuracy = predict_binary(train_x, train_y, parameters)
pred_test, test_accuracy = predict_binary(test_x, test_y, parameters)
print("test_y", test_y[:, :20])
print("pred_y", pred_test[:, :20])
print("Train Accuracy = ", train_accuracy)
print("Test  Accuracy = ", test_accuracy)

#plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (hundred)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


