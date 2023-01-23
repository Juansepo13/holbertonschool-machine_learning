#!/usr/bin/env python3
"""
File: 8-transformer_decoder_block.py
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    class Encoder that inherits from tensorflow.keras.layers.Layer
    to create the encoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Constructs all weights for the model, call the model with an input to
        build the model
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
          - hidden is the number of hidden units in the fully connected layer
          - input_vocab is an integer representing the size of the input
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
          - blocks - a list of length N containing all of the EncoderBlock's
          - dropout - the dropout layer, to be applied to the positional
            encodings
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Method call should use tf.cast to convert mask to tf.float32
        Arguments:
          - x is a tensor of shape (batch, input_seq_len, dm) containing the
              input to the encoder
          - training is a boolean to determine if the model is training
          - mask is the mask to be applied for multi head attention
        Returns:
          - A tensor of shape (batch, input_seq_len, dm) containing the
            encoder output
        """
        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        output = self.dropout(embedding, training=training)
        for i in range(self.N):
            output = self.blocks[i](output, training, mask)
        return output
