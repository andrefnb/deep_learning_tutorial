#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:46:01 2020

@author: andrebastos
"""

#%tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow_tutorial
import keras_sequential_tutorial
import keras_functional_tutorial

# Avocado Prices it's a dataset containing more than 10000 observations and 13 variables.
# Since it's size goes beyong 2MB I did some pre-processing in order to get a fair sample under 1MB. I did this using the function sample from pandas library, as it also shuffles before sampling the data.
# Everyone knows that Millenials love avocado toasts, but it is not cheap at all. The goal of this analysis is to understand in what USA region it's cheaper to eat attending to Millenial standards.
# So we are trying to predict the average price of avocados, being the AveragePrice our targets.
# It is a linear regression problem

# Resets 
tf.compat.v1.reset_default_graph()

# Get pre-processed data
avocados = pd.read_csv('filtered_avocados.csv')

# Split date into day and month since integers are alot easier to deal with than categorical values, as we'll see
new_columns = pd.DataFrame(avocados["Date"].str.split('-').tolist(), columns = ['yr', 'month', 'day'])

# Change data type of new columns from str to int32
new_columns["day"] = new_columns["day"].astype(np.int32)
new_columns["month"] = new_columns["month"].astype(np.int32)
# Drop year since our data already has that column
del new_columns['yr']
# Attach new columns
avocados["day"] = new_columns["day"]
avocados["month"] = new_columns["month"]

# Get One-hot encoding to type and region
# This returns us as many columns as there are different data rows regarding that categorical column.
# Each of this columns have a 1 if that observation column corresponds to that particular option, and a 0 if it doesn't.
# As this adds one dimension for every option answered for the categorical column, having too many dimensions
# might lead us into the "curse of dimensionality" problem and overfitting. But it is still better than just index every answer to a number
# and represent options as it's designated index, since that presupposes a scale we do not wish to represent.
avocados["type"]=pd.Categorical(avocados["type"])
avocados["region"]=pd.Categorical(avocados["region"])
df_type = pd.get_dummies(avocados['type'], prefix = 'category')
df_region = pd.get_dummies(avocados['region'], prefix = 'category')

# Drop Date, type and region
del avocados["Date"]
del avocados["type"]
del avocados["region"]

# Attach the one-hot encodings columns to other integer ones
avocados_cleaned=pd.concat([avocados, df_type, df_region], axis=1, sort=False)

# Shuffle data using the pandas function sample, with fraction as 1 since we want all data
avocados = avocados_cleaned.sample(frac=1)

# Converto to numpy array
avocados = avocados.to_numpy()

# Get the mean and standard deviation for standardization
means = np.mean(avocados,axis=0)
stds = np.std(avocados,axis=0)
# Standardize
avocados = (avocados-means)/stds
# Standard deviation and mean from labels for later reconvertion from error to currency
std_Ys = stds[0]
mean_Ys = means[0]
# Get the labels
Ys = avocados[:, [0]]
# Get the data without the labels
Xs = avocados[:,1:]

# Get training and validation sets
test_Y = Ys[7000:,]
test_X = Xs[7000:,]
valid_Y = Ys[6000:7000,]
valid_X = Xs[6000:7000,]
Y = Ys[:6000,]
X = Xs[:6000,]

# Quadratic error
def mean_squared_error(predicted, y):
    cost = tf.reduce_mean(tf.math.square(y-predicted))
    return cost

# Function that reconverts the errors obtained in currency again - power them and then multiply by standard deviation for label value
def reconvert_to_currency(list, std, power=0.5): 
    return [number**power*std for number in list]

# Function that plots the error graphics
def plot_errors(train_errors, val_errors):
    f, ax = plt.subplots(1)
    ax.plot(train_errors,'b',label='Training loss')
    ax.plot(val_errors,'r',label='Validation loss')
    ax.legend()
    plt.show(f)
    
# Function that shows comparation between true values and predicted ones
def compare_predictions(test_predicted, test_Y):
    columns=['Model prediction', "True values"]
    comparison_predictions = np.zeros([300,2])
    comparison_predictions[:,[0]] = (test_predicted * std_Ys) + mean_Ys
    comparison_predictions[:,[1]] = (test_Y * std_Ys) + mean_Ys
    df=pd.DataFrame(data=comparison_predictions, columns=columns)
    print(df)

def train_tensorflow_net():
    
    # Define network architecture
    layers = [50,25,1]
    # Instantialize Network object
    reg = tensorflow_tutorial.Tensorflow_Regression(train_inputs = X, train_labels = Y, val_input = valid_X, val_labels = valid_Y, std = std_Ys)
    # Train the network
    train_errors, val_errors, network = reg.train(epochs = 100, layers = layers)
    # Plot history - doesn't need reconvertion because it already happens inside training
    plot_errors(train_errors, val_errors)
    
    # Predict the test set
    test_predicted = reg.predict(tf.constant(test_X.astype(np.float32)), network)
    # Get test error
    test_error = mean_squared_error(test_predicted, test_Y)
    # Reconvert to a meaningful scale
    test_error = test_error ** 0.5 * std_Ys
    print(f"Test error: {test_error}")
    
    compare_predictions(test_predicted, test_Y)

def train_keras_sequential():
    
    # Instantialize Network object
    reg = keras_sequential_tutorial.Sequential_Regression(train_inputs = X, train_labels = Y, val_input = valid_X, val_labels = valid_Y)
    # Train the network
    history, model = reg.train(epochs = 300)
    history_dict = history.history
    train_error = history_dict['loss']
    val_error = history_dict['val_loss']
    # Reconvert errors into currency for additional meaning
    train_error = reconvert_to_currency(train_error, std = std_Ys)
    val_error = reconvert_to_currency(val_error, std = std_Ys)
    
    # Plot history
    plot_errors(train_error, val_error)
    # Predict the test set
    test_error = model.evaluate(test_X, test_Y)
    # Reconvert to a meaningful scale
    test_error = test_error ** 0.5 * std_Ys
    print(f"Test error: {test_error}")
    
    test_predicted = model.predict(test_X)
    compare_predictions(test_predicted, test_Y)
    
def train_keras_functional():
    
    # Instantialize Network object
    reg = keras_functional_tutorial.Functional_Regression(train_inputs = X, train_labels = Y, val_input = valid_X, val_labels = valid_Y)
    # Train the network
    history, model = reg.train(epochs = 300)
    history_dict = history.history
    train_error = history_dict['loss']
    val_error = history_dict['val_loss']
    # Reconvert errors into currency for additional meaning
    train_error = reconvert_to_currency(train_error, std = std_Ys)
    val_error = reconvert_to_currency(val_error, std = std_Ys)
    
    # Plot history
    plot_errors(train_error, val_error)
    # Predict the test set
    test_error = model.evaluate(test_X, test_Y)
    # Reconvert to a meaningful scale
    test_error = test_error ** 0.5 * std_Ys
    print(f"Test error: {test_error}")
    
    test_predicted = model.predict(test_X)
    compare_predictions(test_predicted, test_Y)

# To visualize the graphs for training and validation errors in tensor board we need to execute the following command in the directory where there is the log directory
# tensorboard --logdir logs
# Then we go to http://127.0.0.1:6006 and can interatively visualize the graphs
# This is implemented only for tensorflow tutorial

# For all tutorials you can check the error graphic ploted by plot_errors function
train_tensorflow_net()
train_keras_sequential()
train_keras_functional()








