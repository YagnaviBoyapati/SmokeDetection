# input
df = spark.read.option("header","true").option("inferSchema","true").csv("dbfs:/FileStore/tables/smoke_2.csv")

#removing non-useful columns
df2 = df.drop("_c0", "UTC", "CNT").drop_duplicates()
df2.show()

# Using Pandas to implement RNN and splitting the data into Train and Test dataset
import pandas as pd
pd_data=df2.toPandas()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Xtrain, Xtest=df2.randomSplit([0.7,0.3],seed=2018)
xtrain=Xtrain.toPandas()
ytrain=xtrain.iloc[:,-1]
xtrain=xtrain.iloc[:,:-1]
xtest=Xtest.toPandas()
ytest=xtest.iloc[:,-1]
xtest=xtest.iloc[:,:-1]

# data preprocessing

s_scaler = StandardScaler()
s_scaler.fit(xtrain)
xtrain = s_scaler.transform(xtrain)
xtest = s_scaler.transform(xtest)

ytest_arr=ytest.values
ytest_arr

xtest_arr=[(x,y)for x,y in zip(xtest,ytest_arr)]
xtest=xtest_arr

#Define layers and no of nodes per layer 
import numpy as np
#[inputs]+hidden_layers+[outputs] # [12,8,6,9,1]
#Initialize weights, activations, derivatives
weights=[]
activations=[]
derivatives=[]
def initialization(layers=[12,4,6,2,1]):
    global weights
    global activations
    global derivatives
    weights=[]
    activations=[]
    derivatives=[]
    for i in range(len(layers) - 1):
    #weights
        w = np.random.uniform(low=0.1, high=0.5, size=(layers[i], layers[i + 1]))
        weights.append(w)
    #derivatives
        d = np.zeros((layers[i], layers[i + 1]))
        derivatives.append(d)
    for i in range(len(layers)):
        a = np.zeros(layers[i])
        activations.append(a)

#Forward Propogation
def forward_propagate(inputData):

    global activations
    global weights

    act = inputData

    activations[0] = act

        # Network layer iteration
    for index, weight in enumerate(weights):
            # Matrix multiplication between previous activation and weight matrix
        act = sigmoid(np.dot(act, weight))
    
        activations[index + 1] = act
    return act


# Define Sigmoid function

def sigmoid(x):
    y = 1.0 / (1 + np.exp(-x))
    return y


# Define Sigmoid Derivative function

def sigmoid_derivative(x):
    return x * (1.0 - x)


# Define Backward propogation
def back_propagate(error):

    global derivatives
    global activations
    global weights

    for i in reversed(range(len(derivatives))):

            # Activation - previous layer
        act = activations[i+1]

            # Sigmoid derivative function
        delta_value = error * sigmoid_derivative(act)

            # reshape delta
        delta_reshape = delta_value.reshape(delta_value.shape[0], -1).T


            # Activations - current layer
        current_layer_activations = activations[i]

            # Reshape activations
        current_layer_activations = current_layer_activations.reshape(current_layer_activations.shape[0],-1)

            
            # Matrix multiplication
        derivatives[i] = np.dot(current_layer_activations, delta_reshape)

            # backpropogate the next error
        error = np.dot(delta_value, weights[i].T)


# Define Gradient Descent
def gradient_descent(learning_rate=1):
    global derivatives
    global weights
    for i in range(len(weights)):
        weight = weights[i]
        derivative = derivatives[i]
        weight += derivative * learning_rate
        weights[i] = weight


#Make Predictions

def predict(X):
    return np.round(forward_propagate(X))


# Define MSE
 
def mse(target_value, output):
    return np.average((target_value - output) ** 2)


#inputs= xtrain, targets = ytrain
def train(X, Y, layers = [12,4,6,2,1], epochs = 5, learning_rate = 1):
        mse_errors = []
        initialization(layers)
        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(X):
                target_value = Y[j]

                # getting output using forward propagation
                output = forward_propagate(input)
               
                error = target_value - output
                back_propagate(error)
                
                # now perform gradient descent on the derivatives
                # (this will update the weights)
                gradient_descent(learning_rate)
                # keep track of the MSE for reporting later
                sum_errors += mse(target_value, output)

            # Epoch complete, report the training error
            error = sum_errors / len(X)
            print("Error {} at epoch {} ".format(error, i+1))
            mse_errors.append(error)

        print("=====*====Training complete!=====*====")
        return mse_errors

# Performing Grid search 

# this step will take around 8-10 minutes
hl=[ [12,4,1], [12,4,6,1], [12,4,6,2,1]]
lr=[0.1,0.05, 0.03]
ep=[5,10]

rdd = sc.parallelize([(i,j,k) for i in hl for j in ep for k in lr])

# Apply the function on each combination of i, j, and k using map
result_rdd = rdd.map(lambda x: ((x[0],x[1],x[2]),train(xtrain, ytrain, x[0], x[1], x[2]))).cache()
result_rdd.collect()


# main
epochs = 10
layers = [12, 4, 6, 1]
mse_errors = train(xtrain, ytrain, layers, epochs, learning_rate=0.05)

epochs_arr = np.arange(1, epochs + 1) 


import matplotlib.pyplot as plt

# Create the line graph
combine_learning_rate = result_rdd.map(lambda x :((x[0][1], x[0][2]), (x[0][0], x[1]))).groupByKey().map(lambda x: (x[0], list(x[1])))
grid_search_list = combine_learning_rate.collect()
grid_search_list

for grids in grid_search_list:
    x_arr = np.arange(1, grids[0][0]+1)
    y_arr = grids[1]
    plt.plot(x_arr, grids[1][0][1], label='1 hidden layer', color='blue', marker='o', linestyle='-')
    plt.plot(x_arr, grids[1][1][1], label='2 hidden layer', color='green', marker='x', linestyle='--')
    plt.plot(x_arr, grids[1][2][1], label='3 hidden layer', color='red', marker='^', linestyle='-.')


    # Customize the plot (optional)
    plt.title("MSE Graph for learning rate= {}".format(grids[0][1]))
    plt.xlabel("number of epochs")
    plt.ylabel("MSE")

    # Display the graph
    plt.legend()
    plt.show()


xtest_rdd=sc.parallelize(xtest)
output_rdd=xtest_rdd.map(lambda x:((int(predict(x[0])[0]),x[1]),1))

def serial_testing(xtest):
    predictedValue = []
    for index, xtest_ex in enumerate(xtest):
        predictedValue.append(int(predict(xtest_ex[0])))
    return predictedValue
serial_testing(xtest)

performance_metrics = output_rdd.reduceByKey(lambda x,y:x+y)

TP = performance_metrics.lookup((1, 1))[0]
TN = performance_metrics.lookup((0, 0))[0]
FP = performance_metrics.lookup((1, 0))[0]
FN = performance_metrics.lookup((0, 1))[0]

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
print("Accuracy of the test data = " + str(Accuracy))
print("Precision of the test data = " + str(Precision))
print("Recall of the test data = " + str(Recall))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(TP, TN, FP, FN):
    # Create the confusion matrix as a 2x2 NumPy array
    confusion_matrix = np.array([[TN, FP], [FN, TP]])
    
    # Set up the plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(TP, TN, FP, FN)