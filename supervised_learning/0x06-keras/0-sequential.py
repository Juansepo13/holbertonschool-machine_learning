#!/usr/bin/env python3
'''Sequential module'''
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    '''builds a neural network with the Keras library
    Args:
        nx is the number of input features to the network
        layers is a list containing the number of nodes in each layer of the
               network
        activations is a list containing the activation functions used for
                    each layer of the network
        lambtha is the L2 regularization parameter
        keep_prob is the probability that a node will be kept for dropout
    Returns: the keras model
    '''
    w = K.initializers.VarianceScaling(mode="fan_avg")
    regularizer = K.regularizers.l2(lambtha)
    model = K.Sequential(name="my_sequential")
    for i in range(len(layers)):
        if i == 0:
            i_shape = (nx,)
        else:
            i_shape = (model.layers[i - 1].output_shape[0],)
        layer = K.layers.Dense(units=layers[i],
                               activation=activations[i],
                               kernel_initializer=w,
                               kernel_regularizer=regularizer,
                               input_shape=i_shape)
        dropout = K.layers.Dropout(1 - keep_prob,
                                   input_shape=i_shape)
        model.add(layer)
        if i < len(layers) - 1:
            model.add(dropout)
    return model
