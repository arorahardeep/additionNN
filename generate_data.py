#!/usr/bin/env python

"""
This module generates data for addition approximation
@author : Hardeep Arora
@date   : 27 Sep 2017
"""

import numpy as np
import pandas as pd
import keras

class GenData:
    _train_per_cat = 2250
    _n_classes = 201
    _train_sz = _train_per_cat * _n_classes
    _test_sz=10000

    def _rm(self, x1, x2, result_train, val):
        z1 = np.where(x1==val)
        x1 = np.delete(x1, z1)
        x2 = np.delete(x2, z1)
        result_train = np.delete(result_train, z1)

        z2 = np.where(x2==val)
        x1 = np.delete(x1, z2)
        x2 = np.delete(x2, z2)
        result_train = np.delete(result_train, z2)

        return x1, x2, result_train

    def _shuffle_in_unison(self,a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def generate_train(self):
        """
        This method generates the train data with balanced classes and saves it in data folder
        :return:
            none
        """

        print("Generating training data...")
        # generate a balanced list of results
        result_train = list(range(self._n_classes)) * self._train_per_cat

        # initialize the x1 and x2 arrays
        x1 = np.array(0)
        x2 = np.array(0)

        # generate random nos that add up to the result_train
        for n1 in result_train:
            if n1 == 0:
                x1 = np.append(x1, 0)
                x2 = np.append(x2, 0)
            else:
                x  = np.random.randint(low = n1/2, high = min(n1,101), size=1)
                x1 = np.append(x1, x)
                x  = n1 - x
                x2 = np.append(x2, x)

        # delete the dummy first entry
        x1 = np.delete(x1, (0), axis=0)
        x2 = np.delete(x2, (0), axis=0)

        # normalize x1 and x2
        x1 = x1/100
        x2 = x2/100
        result_train = np.array(result_train)

        # remove a number from training x and y in this case we remove 50
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.5)
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.29)
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.05)

        # print debug statement
        print(x1[np.argmax(x2)])
        print(x2[np.argmax(x2)])

        # create the training data
        x_train = np.dstack((x1,x2))

        x_train = x_train.reshape(x_train.shape[1],x_train.shape[2])

        self._shuffle_in_unison(x_train, result_train)

        print("Testing data integrity after shuffle...")
        print(x_train[52], result_train[52])
        print(x_train[152], result_train[152])

        y_train = keras.utils.to_categorical(result_train, self._n_classes)

        print("train_x shape = %s , train_y shape = %s" % (x_train.shape, y_train.shape))

        train_x = pd.DataFrame(data=x_train, columns=['X1','X2'])
        train_x.to_csv("data/add_train_x.csv",header=True,index=False)
        train_y = pd.DataFrame(data=y_train)
        train_y.to_csv("data/add_train_y.csv",header=True,index=False)

    def generate_test(self):
        """
        This method generates the test data and saves it in data folder
        :return:
            none
        """

        print("Generate test data...")
        x1 = np.random.randint(101, size=self._test_sz)
        x2 = np.random.randint(101, size=self._test_sz)

        result_test = x1 + x2

        x1 = x1/100
        x2 = x2/100

        x_test = np.dstack((x1,x2))
        x_test = x_test.reshape(x_test.shape[1],x_test.shape[2])

        y_test = keras.utils.to_categorical(result_test, self._n_classes)

        print("test_x shape = %s , test_y shape = %s" % (x_test.shape, y_test.shape))

        train_x = pd.DataFrame(data=x_test, columns=['X1','X2'])
        train_x.to_csv("data/add_test_x.csv",header=True,index=False)
        train_y = pd.DataFrame(data=y_test)
        train_y.to_csv("data/add_test_y.csv",header=True,index=False)

    def generate_train_1(self):
        """
        This method generates the train data and saves it in data folder
        :return:
            none
        """

        print("Generate train data...")
        x1 = np.random.randint(101, size=self._train_sz)
        x2 = np.random.randint(101, size=self._train_sz)

        result_train = x1 + x2

        x1 = x1/100
        x2 = x2/100

        # remove a number from training x and y in this case we remove 50
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.5)
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.29)
        x1, x2, result_train = self._rm(x1, x2, result_train, 0.05)

        x_train = np.dstack((x1,x2))
        x_train = x_train.reshape(x_train.shape[1],x_train.shape[2])

        y_train = keras.utils.to_categorical(result_train, self._n_classes)

        print("train_x shape = %s , train_y shape = %s" % (x_train.shape, y_train.shape))

        train_x = pd.DataFrame(data=x_train, columns=['X1','X2'])
        train_x.to_csv("data/add_train_x.csv",header=True,index=False)
        train_y = pd.DataFrame(data=y_train)
        train_y.to_csv("data/add_train_y.csv",header=True,index=False)

    @staticmethod
    def generate():
        """
        This method generates the data and saves it in data folder
        :return:
            none
        """
        g = GenData()
        g.generate_train()
        g.generate_test()

    @staticmethod
    def load_data():
        """
        This method loads the train and test data and returns it
        :return:
            x_train - training data
            x_test  - testing  data
            y_train - training labels
            y_test  - testing  labels
        """

        x_train = pd.read_csv("data/add_train_x.csv")
        y_train = pd.read_csv("data/add_train_y.csv")

        x_train = x_train.values
        y_train = y_train.values

        print("train_x shape = %s , train_y shape = %s" % (x_train.shape, y_train.shape))

        x_test = pd.read_csv("data/add_test_x.csv")
        y_test = pd.read_csv("data/add_test_y.csv")

        x_test = x_test.values
        y_test = y_test.values

        print("test_x shape = %s , test_y shape = %s" % (x_test.shape, y_test.shape))

        print("Random nos pair: "+ str(x_train[101])+ "," + str(np.argmax(y_train[101])))

        return x_train, x_test, y_train, y_test

def main():
    #GenData.load_data()
    GenData.generate()

if __name__ == "__main__":
    main()
