#!/usr/bin/env python

"""
This module generates data for addition approximation
@author : Hardeep Arora
@date   : 27 Sep 2017
"""

from keras.layers import *
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from generate_data import GenData as gd
from keras.optimizers import Adam

class AddApproxModel:

    _learning_rate = 0.5e-5
    _decay_rate = 0.1e-6
    _training_epochs = 2500
    _batch_size = 128
    _n_input = 2      # X1 and X2 are the nos to be added
    _n_hidden_1 = 128 # 1st layer number of neurons
    _n_hidden_2 = 256 # 2nd layer number of neurons
    _n_hidden_3 = 512 # 2nd layer number of neurons
    _n_hidden_4 = 256 # 2nd layer number of neurons
    _n_classes = 201
    _model = None
    _hist = None
    model_file_name = "models/nn_add_approx_09"

    def build_model(self):
        Inp = Input(shape=(self._n_input,) )
        x = Dense(self._n_hidden_1, activation='relu', name = "Dense_1")(Inp)
        #x = Dropout(0.2)(x)
        x = Dense(self._n_hidden_2, activation='relu', name = "Dense_2")(x)
        x = Dense(self._n_hidden_3, activation='relu', name = "Dense_3")(x)
        x = Dropout(0.3)(x)
        x = Dense(self._n_hidden_4, activation='relu', name = "Dense_4")(x)

        output = Dense(self._n_classes, activation='softmax', name = "Outputlayer")(x)

        self._model = Model(Inp, output)

    @staticmethod
    def _set_checkpoint():
        filepath="checkpoints/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        return callbacks_list

    def train(self, x_train, y_train, x_test, y_test):

        self._model.compile(loss='categorical_crossentropy',
                            optimizer=Adam(lr=self._learning_rate,decay=self._decay_rate),
                            metrics=['accuracy'])

        self._model.summary()

        history = self._model.fit(x_train, y_train,
                    batch_size=self._batch_size,
                    epochs=self._training_epochs,
                    #callbacks=AddApproxModel._set_checkpoint(),
                    verbose=1, # This is for what we want it to display out as it trains
                    validation_data=(x_test, y_test))

        self._hist = history

    def plot_train(self):
        h = self._hist.history
        if 'acc' in h:
            meas='acc'
            loc='lower right'
        else:
            meas='loss'
            loc='upper right'
        plt.plot(self._hist.history[meas])
        plt.plot(self._hist.history['val_'+meas])
        plt.title('model '+meas)
        plt.ylabel(meas)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc=loc)
        plt.show()

    def plot_train_1(self):
        h = self._hist.history
        print(h)

    def save_model(self, model_name):
        # serialize model to JSON
        model_json = self._model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self._model.save_weights(model_name + ".h5")
        print("Saved model to disk")

    def load_model(self, model_name):
        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json)
        # load weights into new model
        self._model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def predict(self, x_pred):
        pred = self._model.predict(x_pred)
        return np.argmax(pred)


def train_model():
    x_train, x_test, y_train, y_test = gd.load_data()
    model_add_approx = AddApproxModel()
    model_add_approx.build_model()
    model_add_approx.train(x_train,y_train,x_test,y_test)
    model_add_approx.plot_train_1()
    model_add_approx.save_model(AddApproxModel.model_file_name)
    return model_add_approx


def load_model():
    model_add_approx = AddApproxModel()
    model_add_approx.load_model(AddApproxModel.model_file_name)
    return model_add_approx


def gen_pred_set():
    print("Enter value for X1: ")
    x1 = input()

    print("Enter value for X2: ")
    x2 = input()

    x1_pred = np.array(int(x1)/100)
    x2_pred = np.array(int(x2)/100)
    x_pred = np.dstack((x1_pred,x2_pred))
    x_pred = x_pred.reshape(x_pred.shape[1],x_pred.shape[2])
    return x_pred


def train_or_predict(mode):
    if mode == "train":
        train_model()
    else:
        model = load_model()
        print("Addition Approx = %d" %model.predict(gen_pred_set()))


def main():
    train_or_predict("train")


if __name__ == "__main__":
    main()

