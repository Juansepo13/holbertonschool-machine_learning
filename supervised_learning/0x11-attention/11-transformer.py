#!/usr/bin/env python3
"""
File Name: 3-gensim_to_keras.py
"""

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    class Transformer that inherits from tensorflow.keras.Model
    to create a transformer network
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input
        to build the model
        Arguments:
        - dm is an integer representing the dimensionality of the model
        - h is an integer representing the number of heads
        - hidden is the number of hidden units in the fully connected layer
        - target_vocab is an integer representing the size of the target
          vocabulary
        - max_seq_len is an integer representing the maximum sequence length
          possible
        - drop_rate is the dropout rate
        Public instance attributes:
        - N - the number of blocks in the encoder
        - dm - the dimensionality of the model
        - embedding - the embedding layer for the inputs
        - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
        - blocks - a list of length N containing all of the EncoderBlock's
        - dropout - the dropout layer, to be applied to the positional
          encodings
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - inputs is a tensor of shape (batch, input_seq_len) containing the
          - target is a tensor of shape (batch, target_seq_len) containing the
          - training is a boolean to determine if the model is training
          - encoder_mask is the padding mask to be applied to the encoder
          - look_ahead_mask is the look ahead mask to be applied to the decoder
          - decoder_mask is the padding mask to be applied to the decoder
        Returns:
          - A tensor of shape (batch, target_seq_len, target_vocab) containing
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
