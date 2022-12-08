#!/usr/bin/env python3
"""
3. Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder
    Args:
        input_dims: int containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each
            hidden layer in the encoder,
        latent_dims: int containing the dimensions of the latent
            space representation

    Returns: encoder, decoder, auto
    """
    k = keras.layers
    K = keras.backend
    Loss = keras.losses.binary_crossentropy

    def sampling(args):
        """Samples similar points from latent space"""
        mean, log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(mean)[0], latent_dims), mean=0, stddev=0.1)
        return mean + K.exp(log_sigma) * epsilon

    input_layer = keras.Input(shape=input_dims)
    coded_input = keras.Input(shape=latent_dims)

    encoded_layers = k.Dense(hidden_layers[0], activation="relu")(input_layer)

    for nodes in hidden_layers[1:]:
        encoded_layers = k.Dense(nodes, activation="relu")(encoded_layers)

    encoded_layers = k.Dense(latent_dims)(encoded_layers)
    encode_mean = k.Dense(latent_dims)(encoded_layers)
    encode_log = k.Dense(latent_dims)(encoded_layers)
    Z = k.Lambda(sampling)([encode_mean, encode_log])

    decoded = k.Dense(hidden_layers[-1], activation="relu")(coded_input)
    for dim in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(dim, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(input_layer, [encode_mean, encode_log, Z])
    decoder = keras.Model(coded_input, decoded)
    outputs = decoder(encoder(input_layer)[-1])
    autoencoder = keras.Model(input_layer, outputs)

    reconstruction_loss = Loss(input_layer, outputs)
    reconstruction_loss *= input_dims
    lat_loss = 1 + encode_log - K.square(encode_mean) - K.exp(encode_log)
    lat_loss = K.sum(lat_loss, axis=-1)
    lat_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + lat_loss)
    autoencoder.add_loss(vae_loss)
    autoencoder.compile(optimizer="adam")

    return encoder, decoder, autoencoder
