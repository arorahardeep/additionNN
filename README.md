## Synopsis

This is a toy example of approximating addition function using neural networks. The input is two numbers between 0-100 and target is a class between 0-200. The network approximates addition function by looking a several thousand such examples.

## Installation

This is written using python 3.5 version. 

> python generate_data.py

Using TensorFlow backend.
Generating training data...
0.99
1.0
Testing data integrity after shuffle...
[ 0.78  0.69] 147
[ 0.62  0.42] 104
train_x shape = (425571, 2) , train_y shape = (425571, 201)
Generate test data...
test_x shape = (10000, 2) , test_y shape = (10000, 201)


This would generate the training and test data under the data folder

>> data/add_train_x.csv  - train data (2250x201,2) - contains randomly generated number pairs between 0-100 normalized between 0-1.
>> data/add_test_x.csv   - test  data (10000, 2)   - contains randomly generated number pairs between 0-100 normalized between 0-1.
>> data/add_train_y.csv  - train labels (2250x201, 201) - contains addition results for examples in train set one-hot encoded.
>> data/add_test_y.csv   - test lables(10000, 201)      - contains addition results for examples in test  set one-hot encoded.

We have removed certain numbers from training, so that they are never seen at train time.

> python nn_add_approx.py

This would create a simple nn with 4 dense hidden layers, the last layer is a softmax with 201 classes.



