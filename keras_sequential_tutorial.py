#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:43:26 2020

@author: andrefnb
"""

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dense, LeakyReLU
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

class Sequential_Regression:
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
        
        # Initialize keras sequential standard model 
        model = Sequential()
        # Add three dense layers with an activation afterwards of Leaky Relu
        # It would also make sense to add a batch normalization layer after every non output layer so that
        # the values don't grow too large, altho in this case it worsened the results
        # For weight initialization the default for keras is already the Normalized initializer so we don't need 
        # to change for a better one, but we could specify through the kernel_initializer paramenter in the dense layer
        model.add(Dense(30))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(15))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(10))
        model.add(LeakyReLU(alpha=0.1))
        # Add a one neuron dense layer to output the numerical result
        model.add(Dense(1))
        # Initialize the Adam optimizer with the chosen learning rate and a fit decay rate for the number of epochs
        opt = Adam(lr = learning_rate, decay = learning_rate / epochs)
        # Compile the model with the mean squared error for this regression problem and the adam optimizer initialized earlier
        model.compile(loss="mean_squared_error", optimizer=opt)
        # Initialize an early stopping method so that it returns the model before it overfits too much
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        # Join callbacks into an array
        # Implementation with tensorboard would be joined here
        # These callbacks are called each iteration of the training, in this case it checks for early stopping criteria every epoch
        callbacks = [early_stop]
        # Fit the data to the compiled model choosing the batch size, epoch number and callbacks
        history = model.fit(X, Y, validation_data=(valid_X, valid_Y), batch_size = batch_size, epochs = epochs, callbacks=callbacks)
        # Show model summary - layers, shapes and parameters
        model.summary()
        
        return history, model






