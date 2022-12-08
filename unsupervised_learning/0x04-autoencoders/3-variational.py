#!/usr/bin/env python3
"""
    Variational autoencoder Module.
"""

import tensorflow.keras as keras


def create_encoder(input_layer, hidden_layers, latent_dims):
    """
        Creates an encoder model
        Args:
                input_layer: the input layer of the model
            hidden_layers: list of the number of nodes for each hidden
                        layer in the encoder
            latent_dims: dimensions of the latent space representation
        Returns:
            encoder: the encoder model
    """

    def sampling(args):
        """
            Creates a sampler.
            Args:
                mean: the mean of the latent space representation
                standard_deviation: the standard deviation of the latent space
                                    representation
                latent_dims: dimensions of the latent space representation
            Returns:
                sampler: the sampler
        """

        mean, standard_deviation = args

        standard_normalization = keras.backend.random_normal(
            shape=(keras.backend.shape(mean)[0], latent_dims),
            mean=0.,
            stddev=1.
        )

        return mean + \
            keras.backend.exp(standard_deviation / 2) * \
            standard_normalization

    encoder_hidden_layers = input_layer

    # Create hidden layers
    for layer in hidden_layers:
        encoder_hidden_layers = keras.layers.Dense(
            layer,
            activation='relu'
        )(encoder_hidden_layers)

    mean = keras.layers.Dense(latent_dims)(encoder_hidden_layers)
    standard_deviation = keras.layers.Dense(latent_dims)(encoder_hidden_layers)

    encoder_output_layer = keras.layers.Lambda(
        sampling
    )([mean, standard_deviation])

    # Create the encoder
    return keras.Model(
        inputs=input_layer,
        outputs=[mean, standard_deviation, encoder_output_layer]
    ), mean, standard_deviation


def create_decoder(hidden_layers, latent_dims, input_dims):
    """
        Creates a decoder model
        Args:
            hidden_layers: list of the number of nodes for each hidden
                    layer in the decoder
                latent_dims: dimensions of the latent space representation
            input_dims: dimensions of the model input
        Returns:
            decoder: the decoder model
    """

    decoder_input_layer = keras.layers.Input(shape=(latent_dims,))

    decoder_hidden_layers = decoder_input_layer

    # Create hidden layers
    for layer in reversed(hidden_layers):
        decoder_hidden_layers = keras.layers.Dense(
            layer,
            activation='relu'
        )(decoder_hidden_layers)

    decoder_output_layer = keras.layers.Dense(
        input_dims,
        activation='sigmoid',
    )(decoder_hidden_layers)

    # Create the decoder
    return keras.Model(
        inputs=decoder_input_layer,
        outputs=decoder_output_layer
    )


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        Creates variational autoencoder.
        Args:
            input_dims: dimensions of the model input
            hidden_layers: list of the number of nodes for each hidden
                            layer in the encoder and decoder
            latent_dims: dimensions of the latent space representation
        Returns:
            encoder: the encoder model
            decoder: the decoder model
            autoencoder: the variational autoencoder model
    """

    # Create the input layer
    input_layer = keras.layers.Input(shape=(input_dims,))

    encoder, mean, standard_deviation = create_encoder(
        input_layer, hidden_layers, latent_dims)
    decoder = create_decoder(hidden_layers, latent_dims, input_dims)
    outputs = decoder(encoder(input_layer)[-1])

    # Create the autoencoder
    autoencoder = keras.Model(input_layer, outputs)

    def kullback_leibler_loss(inputs, outputs):
        reconstruction_loss = keras.losses.binary_crossentropy(
            inputs, outputs) * input_dims
        square_mean = keras.backend.square(mean)
        sigma_exponential = keras.backend.exp(standard_deviation)

        kl_loss = 1 + standard_deviation
        kl_loss -= square_mean
        kl_loss -= sigma_exponential
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return keras.backend.mean(reconstruction_loss + kl_loss)

    # Compile the model with adam optimizer and binary crossentropy loss
    autoencoder.compile(
        optimizer='adam',
        loss=kullback_leibler_loss
    )

    return encoder, decoder, autoencoder
