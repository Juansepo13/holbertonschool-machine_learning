#!/usr/bin/env python3
"""
File Name: 10-transformer_decoder.py
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decoder that inherits from tensorflow.keras.layers.Layer to
    create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input
        to build the model
        Arguments:
        - dm is an integer representing the dimensionality of the model
        - h is an integer representing the number of heads
        - hidden is the number of hidden units in the fully connected layer
        - target_vocab is an integer representing the size of the target
          vocabulary
        - max_seq_len is an integer representing the maximum sequence
          length possible
        - drop_rate is the dropout rate
        Public instance attributes:
        - N - the number of blocks in the encoder
        - dm - the dimensionality of the model
        - embedding - the embedding layer for the inputs
        - positional_encoding - a numpy.ndarray of shape (max_seq_len, dm)
          containing the positional encodings
        - blocks - a list of length N containing all of the EncoderBlock‘s
        - dropout - the dropout layer, to be applied to the positional
          encodings
        """
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        super(Decoder, self).__init__()

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - x is a tensor of shape (batch, input_seq_len, dm) containing
              the input to the encoder
          - training is a boolean to determine if the model is training
          - mask is the mask to be applied for multi head attention
        Returns:
          - A tensor of shape (batch, input_seq_len, dm) containing
            the encoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x
