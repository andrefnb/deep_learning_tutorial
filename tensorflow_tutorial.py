#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:47:21 2020

@author: andrebastos
"""

import tensorflow as tf
import numpy as np
from datetime import datetime

class Tensorflow_Regression:
    train_inputs = []
    train_labels = []
    val_input = []
    val_labels = []
    std = 0
    
    def __init__(self, train_inputs, train_labels, val_input, val_labels, std):
        
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.val_input = val_input
        self.val_labels = val_labels
        self.std = std

    # Function that initializes weights and biases for a given layer of the network architecture.
    # These weights and biases need to be random so that we don't end up with equal weights.
    # Receives the inputs of the layer (might be the data in the case of the first layer 
    # or the weights of previous layers) for shape identification - number of variables.
    # Also receives the number of neurons in the layer.
    def layer(self, inputs,neurons,layer_name):
        # We need to initialize the weights with the constraint of a standard deviation so that it does not take very high values. If weights initialize as very high numbers,
        # we can end up with very large outputs from the neurons and as we compute the quadratic error it gives us a very large error, leading to a very steep gradient that threatens 
        # the convergion of the stochastic gradient descent (it might start to diverge).
        weights = tf.Variable(tf.random.normal((inputs.shape[1],neurons), stddev = 1/neurons ), name = layer_name)
        # With bias we don't need to worry about this as it is used mainly to recenter the function line to the origin
        bias = tf.Variable(tf.zeros([neurons]), name = layer_name)
        return weights,bias
    
    # Function that receives the inputs and network architecture and initializes every layer.
    # Outputs network as an array that joins weights to corresponding biases.
    # Also outputs the same array but with every element (weight or bias) in the same dimension as the rest.
    def create_network(self, X,layers):
        network = []
        variables = []
        # First previous are the data inputs.
        previous = X
        for ix, neurons in enumerate(layers):
            weights,bias = self.layer(previous,neurons,f'layer_{ix}')
            network.append( (weights,bias) )
            variables.extend( (weights,bias) )
            # Chain weights to next layer.
            previous = weights
        return network, variables
    
    # Function that will iterate through the network architecture and apply the linear (multiplication of the weight and sum of the bias) 
    # and non linear (apply the Leaky ReLU activation) tranformations.
    # We could be using the sigmoid activation but it may have the problem of vanishing gradients:
    # When we chain several calculations of derivatives, we can find a derivative that is close to zero. Since the calculations are all products the whole
    # chain can start outputting values closer to zero until it disappears. And when gradients are close to zero then we have less information to update the weights,
    # leading to a slow down of the training only to eventually stop.
    # Leaky ReLU solves this problem by returning the input value if it's greater than zero and returning an arbitrary value based on the input (x/a) if x (input) is less than zero.
    # By return a negative value on the negative side (instead of just 0 like standard ReLU) it avoids another problem. It no longer returns zero values and the gradient won't become zero,
    # avoiding the "death" of the neuron. A dead neuron won't learn anything at all.
    def predict(self, X, network):
        net = X
        layer = 1
        # For every layer except the output one (last one) apply the linear transformation and activation
        for weights,bias in network[:-1]:
            with tf.name_scope(f'Layer_{layer}'):
                net = tf.add(tf.matmul(net, weights), bias,name='net')
                net = tf.nn.leaky_relu(net, name="relu")
            layer += 1
        weights,bias = network[-1]
        # In the output layer we are not interested in what the activation returns so we only transform it linearly.
        # Also as this is a regression case the last layer will be composed of one neuron since we want it to output a single value that is being predicted.
        # This won't be the case for other types of problems.
        with tf.name_scope('Output'):
            net = tf.add(tf.matmul(net, weights), bias)
        return net
    
    # Loss function that will calculate the quadratic error between a predicted value and it's target.
    def mean_squared_error(self, predicted, y):
        cost = tf.reduce_mean(tf.math.square(y-predicted))
        return cost
    
    # Function that will create the GradientTape object that will trace the computations and compute the derivatives.
    # Receive variables because they already are in a list (variable, gradient)
    def grad(self, X, Y, network, variables):
        with tf.GradientTape() as tape:
            predicted = self.predict(X, network)
            loss_val = self.mean_squared_error(predicted,Y)
        return tape.gradient(loss_val, variables),variables
    
    # Decorator that defines the function to be converted into Tensorflow graph code, this function defines the computations to be traced.
    @tf.function
    def create_graph(self, X, network):
        _ = self.predict(X, network)
        
    def write_graph(self, X, writer, network):
        tf.summary.trace_on(graph=True)
        self.create_graph(tf.constant(X.astype(np.float32)), network)
        with writer.as_default():
            tf.summary.trace_export(name="trace",step=0)
            
    # Function that will initiate the training of the network
    # It takes some parameters to specify hyperparameters of the network such as learning rate, batch size, the network structure, epochs
    def train(self, epochs = 50, batch_size = 16, learning_rate = 0.0001, layers=[25,10,1]):
        # Define log directory for tensorboard
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "logs"
        log_dir = "{}/model-{}/".format(root_logdir, now)
        
        X = self.train_inputs
        Y = self.train_labels
        valid_X = self.val_input
        valid_Y = self.val_labels
        std = self.std
        
        # Get the network layer initialization
        network, variables = self.create_network(X,layers)
        # Calculate the batches per epoch
        batches_per_epoch = X.shape[0]//batch_size
        # Initialize the optimizer to be used, could be Stochastic Gradient Descent with a specific learning rate and momentum, that cannot be too high so it won't
        # overstep and go beyond the optimal solution.
        # This type of optimizer has the problem of having a constant learning rate, being unable to adapt to different conditions along minimization.
        # So if a gradient is flat then the LR might still be low and hold back the learning, making further slow improvements
        # So we will utilize the Adam optimizer. Just like Adagrad, this optimizer keeps track of the sum of the squared gradients for all parameters,
        # computing a learning rate more appropriate for every parameter. Parameters with small gradients will have large LR and gradients with larger gradients
        # will have small LR
        # This Adam optimizer also adapts the momentum for every situation and utilizes an exponentially decaying average over the previous gradients.
        # It is usually the fastest choice, being careful for not going beyond the optimal solution.
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)#, momentum = 0.9) SGD
        
        # Create file writter for tensorboard
        writer = tf.summary.create_file_writer(log_dir)
        # Create computation graph
        self.write_graph(X, writer, network)
        
        # Vector that will hold training errors for each epoch
        train_errors = []
        # Vector that will hold validation errors for each epoch
        val_errors = []
        
        # Iterate through the epochs
        for epoch in range(epochs):
            # Get indexes for batch row choice
            # It needs to be randomized so that each batch take always a different sample of the data in order to propperly generalize
            # This batch learning consists of the summing the weights updates for a set of examples and then updating the weights with the total change
            shuffled = np.arange(len(Y))
            # Randomize these indexes
            np.random.shuffle(shuffled)
            # Iterate through the batches
            for batch_num in range(batches_per_epoch):
                # Get batch starting position
                start = batch_num*batch_size
                # Get training inputs batch and labels input batch, also turn them into float32 as it is numerically safer
                batch_xs = tf.constant(X[shuffled[start:start+batch_size],:].astype(np.float32))
                batch_ys = tf.constant(Y[shuffled[start:start+batch_size]].astype(np.float32))
                # Go get the gradients to be computed
                gradients,variables = self.grad(batch_xs, batch_ys, network, variables)
                # Compute the gradients using the optimizer and update network
                optimizer.apply_gradients(zip(gradients, variables))
                
            # To get the errors we need to compare the predictions from training and the ones from the test set
            # Predict the labels for the whole input training list 
            train_pred = self.predict(tf.constant(X.astype(np.float32)), network)
            # Predict the labels for the whole input validation list 
            val_pred = self.predict(tf.constant(valid_X.astype(np.float32)), network)
            
            # Compute the quadratic error for train and test predicted sets
            train_error = self.mean_squared_error(train_pred, Y)
            val_error = self.mean_squared_error(val_pred, valid_Y)
            
            # Reconvert the errors into currency for amore meaningful analysis
            # Utilize the standard deviation previously calculated based on the inputs
            train_error = train_error ** 0.5 * std
            val_error = val_error ** 0.5 * std
            
            train_errors.append(train_error)
            val_errors.append(val_error)
            
            # Print results
            print(f"Epoch {epoch}, train error {train_error}, valid error {val_error}")
            # Write results to tensorboard so that we can visualize them
            with writer.as_default():
                tf.summary.scalar('Train error', train_error, step=epoch)
                tf.summary.scalar('Validation error', val_error, step=epoch)
                
        writer.close()
        return train_errors, val_errors, network



