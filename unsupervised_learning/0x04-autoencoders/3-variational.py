#!/usr/bin/env python3
'''Variational Autoencoder module'''
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    '''creates an autoencoder
    Args:
        input_dims is an integer containing the dimensions of the model input
        hidden_layers is a list containing the number of nodes for each hidden
                      layer in the encoder, respectively
            * the hidden layers should be reversed for the decoder
        latent_dims is an integer containing the dimensions of the latent space
                    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    '''
    inputs = keras.Input(shape=(input_dims,))
    layer_enc = keras.layers.Dense(
        hidden_layers[0],
        activation='relu',
    )(inputs)

    len_hl = len(hidden_layers)
    l_hl = len(hidden_layers) - 1
    for i in range(1, len_hl):
        layer_enc = keras.layers.Dense(
            hidden_layers[i],
            activation='relu',)
        (layer_enc)

    z_mean = keras.layers.Dense(latent_dims)(layer_enc)
    z_log_sigma = keras.layers.Dense(latent_dims)(layer_enc)

    def sample(args):
        '''sample function '''
        z_mean, z_log_sigma = args
        s1 = keras.backend.shape(z_mean)[0]
        s2 = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(s1, s2))

        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    z = keras.layers.Lambda(sample)([z_mean, z_log_sigma])
    encoded_input = keras.Input(shape=(latent_dims,))

    latent = keras.layers.Dense(
        hidden_layers[l_hl],
        activation='relu',
    )(encoded_input)

    flag = 1
    for j in range(len_hl - 2, -1, -1):
        dec = keras.layers.Dense(
            hidden_layers[j],
            activation='relu',
        )(latent if flag else dec)
        flag = 0

    decoded = keras.layers.Dense(
        input_dims,
        activation='sigmoid',
    )(latent if flag else dec)

    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z])
    decoder = keras.Model(encoded_input, decoded)
    outputs = decoder(encoder(inputs)[2])

    vae = keras.Model(inputs, outputs)

    def vae_loss(inputs, outputs):
        '''vae loss function'''
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = (1 + z_log_sigma - keras.backend.square(z_mean) -
                   keras.backend.exp(z_log_sigma))
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    vae.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, vae
