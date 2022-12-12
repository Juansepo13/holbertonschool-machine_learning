#!/usr/bin/env python3
"""
1. Sparse Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
            hidden layer in the encoder,
        latent_dims: int containing the dimensions of the latent
            space representation
        lambtha: regularization parameter used for L1 regularization
            on the encoded output

    Returns: encoder, decoder, auto
    """
    regularizer = keras.regularizers.l1(lambtha)

    k = keras.layers
    input = keras.Input(shape=(input_dims,))
    encodedl = k.Dense(hidden_layers[0], activation='relu')(input)
    for layer in hidden_layers[1:]:
        encodedl = k.Dense(layer, activation='relu')(encodedl)
    encodedl = k.Dense(latent_dims, activation='relu',
                       activity_regularizer=regularizer)(encodedl)
    encoder = keras.Model(input, encodedl)

    coded_input = keras.Input(shape=(latent_dims,))
    decodedl = k.Dense(hidden_layers[-1], activation='relu')(coded_input)
    for dim in hidden_layers[-2::-1]:
        decodedl = k.Dense(dim, activation='relu')(decodedl)
    decodedl = k.Dense(input_dims, activation='sigmoid')(decodedl)
    decoder = keras.Model(coded_input, decodedl)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
