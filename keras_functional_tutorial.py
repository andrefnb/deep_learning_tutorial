#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 01:27:12 2020

@author: andrefnb
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from datetime import datetime

class Functional_Regression:
    train_inputs = []
    train_labels = []
    val_input = []
    val_labels = []
    
    def __init__(self, train_inputs, train_labels, val_input, val_labels):
        
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.val_input = val_input
        self.val_labels = val_labels
        
    def train(self, epochs = 50, batch_size = 16, learning_rate = 0.0001):
        # Assign data to convinient names
        X = self.train_inputs
        Y = self.train_labels
        valid_X = self.val_input
        valid_Y = self.val_labels
        # Define input shape as 67 is the number of columns in the data
        inputs = Input(shape=(67,))
        
        # Add three hidden layers with an activation afterwards of Elu
        hidden = Dense(50)(inputs)
        hidden = Activation('elu')(hidden)
        
        # hidden = BatchNormalization()(hidden)
        # hidden = Dropout(0.3)(hidden)
        hidden = Dense(30)(hidden)
        hidden = Activation('elu')(hidden)
        
        # hidden = BatchNormalization()(hidden)
        # hidden = Dropout(0.3)(hidden)
        hidden = Dense(15)(hidden)
        hidden = Activation('elu')(hidden)
        # Add a one neuron dense layer to output the numerical result
        output = Dense(1)(hidden)
        # Finish model creation with the definition of data shapes with inputs and outputs matrices
        model = Model(inputs=inputs, outputs=output)
        # Initialize an early stopping method so that it returns the model before it overfits too much
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        # Join callbacks into an array
        callbacks = [early_stop]
        # Initialize the Adam optimizer with the chosen learning rate and a fit decay rate for the number of epochs
        opt = Adam(lr = learning_rate, decay = learning_rate / epochs)
        # Compile the model with the mean squared error for this regression problem and the adam optimizer initialized earlier
        model.compile(optimizer = opt, loss = "mean_squared_error")
        # Fit the data to the compiled model choosing the batch size, epoch number and callbacks
        history = model.fit(X, Y, validation_data=(valid_X, valid_Y), batch_size = batch_size, epochs = epochs, callbacks=callbacks)
        # Show model summary - layers, shapes and parameters
        model.summary()
        
        return history, model
    






