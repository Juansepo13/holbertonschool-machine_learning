#!/usr/bin/env python3
"""
0x04. Autoencoders
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:
    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
        layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
        representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
        cross-entropy loss
    All layers should use a relu activation except for the mean and log
        variance layers in the encoder, which should use None, and the last
        layer in the decoder, which should use sigmoid
    """
    inputs = keras.Input(shape=(input_dims,))
    enc = inputs

    for layer_dims in hidden_layers:
        enc = keras.layers.Dense(units=layer_dims, activation="relu")(enc)

    mean = keras.layers.Dense(units=latent_dims)(enc)
    log_sigma = keras.layers.Dense(units=latent_dims)(enc)

    def sampling(args):
        mean, log_sigma = args
        epsilon = keras.backend.random_normal(shape=(keras.backend
                                                     .shape(mean)[0],
                                                     latent_dims),
                                              mean=0, stddev=0.1)
        return mean + keras.backend.exp(log_sigma) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,)
                            )([mean, log_sigma])

    dec_input = keras.Input(shape=(latent_dims,))
    dec = dec_input

    for layer_dims in reversed(hidden_layers):
        dec = keras.layers.Dense(units=layer_dims, activation="relu")(dec)

    dec_output_layer = keras.layers.Dense(units=input_dims,
                                          activation="sigmoid")(dec)

    encoder = keras.Model(inputs=inputs, outputs=[z, mean, log_sigma])
    decoder = keras.Model(inputs=dec_input, outputs=dec_output_layer)

    outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=outputs)

    def custom_loss(inputs, outputs, input_dims, log_sigma, mean):
        """custom loss function"""
        def loss(inputs, outputs):
            """loss of the custom loss"""
            rec_loss = keras.losses.binary_crossentropy(inputs, outputs)
            rec_loss *= input_dims
            kl_loss = (1 + log_sigma - keras.backend
                       .square(mean) - keras.backend.exp(log_sigma))
            kl_loss = keras.backend.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            vae_loss = keras.backend.mean(rec_loss + kl_loss)
            return vae_loss
        return loss

    opt = keras.optimizers.Adam()
    loss = "binary_crossentropy"

    encoder.compile(loss=loss, optimizer=opt)
    decoder.compile(loss=loss, optimizer=opt)
    auto.compile(loss=custom_loss(inputs,
                                  outputs,
                                  input_dims,
                                  log_sigma,
                                  mean),
                 optimizer=opt)

    return encoder, decoder, auto
