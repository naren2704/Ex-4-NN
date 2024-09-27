## ENTER YOUR NAME: NARENDRAN.B
## ENTER YOUR REGISTER NO: 212222240069
## EX. NO.4
## DATE:27/09/24
## Implementation of MLP with Backpropagation for Multiclassification
## Aim:
To implement a Multilayer Perceptron for Multi classification
## Theory
A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method. A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique. MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.

## MLP has the following features:

Ø Adjusts the synaptic weights based on Error Correction Rule

Ø Adopts LMS

Ø possess Backpropagation algorithm for recurrent propagation of error

Ø Consists of two passes

(i)Feed Forward pass
         (ii)Backward pass
Ø Learning process –backpropagation

Ø Computationally efficient method

![image](https://github.com/user-attachments/assets/6e06b8b3-db3e-4d13-bdfa-6505394eeaae)


3 Distinctive Characteristics of MLP:

Ø Each neuron in network includes a non-linear activation function

![image](https://github.com/user-attachments/assets/8921730d-abca-4503-ad89-5cc3c3676b61)

Ø Contains one or more hidden layers with hidden neurons

Ø Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

*Originates at an output neuron

*Propagates backward through the network neuron

*Involves error dependent function in one way or the other

Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

## TWO PASSES OF COMPUTATION:

In the forward pass:

• Synaptic weights remain unaltered

• Function signal are computed neuron by neuron

• Function signal of jth neuron is :
![image](https://github.com/user-attachments/assets/83dbad98-98a9-4656-a7e2-fe508fac650b)
![image](https://github.com/user-attachments/assets/87dfcc3e-a0e1-4acf-b2fc-c5da3236034d)


If jth neuron is output neuron, the m=mL and output of j th neuron is image

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer 
![image](https://github.com/user-attachments/assets/9cf022ef-9761-4757-b013-97d79f2c24b2)


In the backward pass,

• It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

• it changes the synaptic weight by delta rule
![image](https://github.com/user-attachments/assets/e7bdbf49-f83a-43e5-abf4-25059c656c57)


## Algorithm:
1.Import the necessary libraries of python.

2.After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3.Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4.Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%. Normalize our dataset.

5.In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6.Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7.In order to get the predicted values we call the predict() function on the testing data set.

8.Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
arr = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=arr)
print(df.head())
a = df.iloc[:, 0:4]
b = df.select_dtypes(include=[object])
b = df.iloc[:,4:5]
training_a, testing_a, training_b, testing_b = train_test_split(a, b, test_size = 0.25)
myscaler = StandardScaler()
myscaler.fit(training_a)
training_a = myscaler.transform(training_a)
testing_a = myscaler.transform(testing_a)
m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)
m1.fit(training_a, training_b.values.ravel())
predicted_values = m1.predict(testing_a)
print(confusion_matrix(testing_b,predicted_values))
print(classification_report(testing_b,predicted_values))
```
## Output:
![image](https://github.com/user-attachments/assets/107fa33f-f398-4f5d-82aa-cac7f0511606)

![image](https://github.com/user-attachments/assets/1f6f5e69-2d27-44f2-a7b6-647968af2068)

![image](https://github.com/user-attachments/assets/35eccef7-f9fa-4faa-a534-76cdb17444bd)


## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
arr = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = pd.read_csv(url, names=arr)
print(df.head())
a = df.iloc[:, 0:4]
b = df.select_dtypes(include=[object])
b = df.iloc[:,4:5]
training_a, testing_a, training_b, testing_b = train_test_split(a, b, test_size = 0.25)
myscaler = StandardScaler()
myscaler.fit(training_a)
training_a = myscaler.transform(training_a)
testing_a = myscaler.transform(testing_a)
m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)
m1.fit(training_a, training_b.values.ravel())
predicted_values = m1.predict(testing_a)
print(confusion_matrix(testing_b,predicted_values))
print(classification_report(testing_b,predicted_values))
```
## Output:

![image](https://github.com/user-attachments/assets/16824be1-f691-4dd2-8147-f226727a8cc1)

![image](https://github.com/user-attachments/assets/1577a418-fca8-4233-bc62-cbf4ac9fd0fc)

![image](https://github.com/user-attachments/assets/995d6369-2e12-42f0-837a-02a38da22a42)


## Result:
Thus, MLP is implemented for multi-classification using python.
