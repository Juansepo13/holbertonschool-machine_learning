#!/usr/bin/env python3
"""
File Name: 3-variational.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder:
    Arguments:
        - input_dims is an integer containing the dimensions of the model input
        - hidden_layers is a list containing the number of nodes for each
                        hidden layer in the encoder, respectively
            * the hidden layers should be reversed for the decoder
        - latent_dims is an integer containing the dimensions of the latent
                      space representation
    Returns: encoder, decoder, auto
        - encoder is the encoder model, which should output the latent
                  representation, the mean, and the log variance, respectively
        - decoder is the decoder model
        - auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    All layers should use a relu activation except for the mean and log
    variance layers in the encoder, which should use None, and the last
    layer in the decoder, which should use sigmoid
    [Expected]
    Autoencoder model is compiled correctly
    5.44117e+02
    """
    # Encoder
    input_encoder = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0], activation='relu')(input_encoder)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(encoded)

    # Latent space
    mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    log_var = keras.layers.Dense(latent_dims, activation=None)(encoded)

    # Sampling
    def sampling(args):
        """
        Sampling function
        """
        mean, log_var = args
        epsilon = keras.backend.random_normal(
            shape=keras.backend.shape(mean), mean=0., stddev=1.)
        return mean + keras.backend.exp(log_var / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_var])

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(input_decoder)
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    # Encoder model
    encoder = keras.Model(inputs=input_encoder, outputs=[z, mean, log_var])

    # Decoder model
    decoder = keras.Model(inputs=input_decoder, outputs=decoded)

    # Autoencoder model
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)[0]))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto