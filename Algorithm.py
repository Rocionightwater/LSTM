#!/usr/bin/env python3
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np

def lstm(x_train,y_train,x_test,y_test):
    print(x_train.shape)
    epochs = 1
    batch_size = 50

    model = Sequential()
    layers = {'input': 16, 'hidden1': 64, 'output': 1}

    model.add(LSTM(1,
            input_shape=(None,16),
            return_sequences=False))
    #model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation("relu"))

    #start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    #print "Compilation Time : ", time.time() - start
    #return model
    print("Training...")
    model.fit(x_train, y_train,batch_size=batch_size, nb_epoch=epochs)
    print("Predicting...")
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))

    print (y_test)
    print (predicted)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(y_test[:93])
    # plt.plot(predicted[:43])
    # plt.show()
