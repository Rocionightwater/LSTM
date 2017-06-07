#!/usr/bin/env python3
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import TimeDistributed, Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam
import numpy as np

def one_layer_lstm(max_len):
    model = Sequential()
    layers = {'input': 43, 'hidden': 128, 'output': 1}

    model.add(LSTM(layers['hidden'],
        input_shape=(max_len, layers['input']),
        return_sequences=True)
    )
    #model.add(Dropout(0.5))
    
    model.add(TimeDistributed(Dense(
        layers['output'])))
    model.add(Activation("softmax"))
    
    optimizer = Adam(lr=0.1)
    model.compile(loss="binary_crossentropy", optimizer='adam')
    model.summary()
    return model  

def lstm(max_len):
    model = Sequential()
    layers = {'input': 43, 'hidden1': 64, 'hidden2' : 128, 'output': 1}

    model.add(LSTM(layers['hidden1'],
        input_shape=(max_len, layers['input']),
        return_sequences=True))
    model.add(Dropout(0.5))
    
    model.add(LSTM(
        layers['hidden2'],
        return_sequences=False))
    model.add(Dropout(0.5))
    
    model.add(Dense(
        layers['output']))
    model.add(Activation("softmax"))
    
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model  
