#!/usr/bin/env python3
"""
1. Sparse Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    Args:
        input_dims: int containing the dimensions of the model input
        filters: list containing the number of filters for each
            convolutional layer in the encoder
        latent_dims: int containing the dimensions of the latent
            space representation

    Returns: encoder, decoder, auto
    """
    k = keras.layers
    input = keras.Input(shape=input_dims)

    encode = k.Conv2D(filters[0], kernel_size=(3, 3),
                      padding='same', activation='relu')(input)
    encode = k.MaxPool2D(pool_size=(2, 2), padding='same')(encode)
    for fil in filters[1:]:
        encode = k.Conv2D(fil, kernel_size=(3, 3),
                          padding='same', activation='relu')(encode)
        encode = k.MaxPool2D(pool_size=(2, 2),
                             padding='same')(encode)
    encoder = keras.Model(input, encode)

    coded_input = keras.Input(shape=latent_dims)
    decode = k.Conv2D(filters[-1], kernel_size=(3, 3),
                      padding='same', activation='relu')(coded_input)
    decode = k.UpSampling2D(size=(2, 2))(decode)
    decode = k.Conv2D(filters[-2], kernel_size=(3, 3),
                      padding='same', activation='relu')(decode)
    decode = k.UpSampling2D(size=(2, 2))(decode)
    decode = k.Conv2D(filters[0], kernel_size=(3, 3),
                      padding='valid', activation='relu')(decode)
    decode = k.UpSampling2D(size=(2, 2))(decode)
    channel = input_dims[-1]
    decode = k.Conv2D(channel, kernel_size=(3, 3),
                      padding='same', activation='sigmoid')(decode)
    decoder = keras.Model(coded_input, decode)

    auto = keras.Model(input, decoder(encoder(input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
