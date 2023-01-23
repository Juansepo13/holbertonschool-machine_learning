#!/usr/bin/env python3
"""
File: 8-transformer_decoder_block.py
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    DecoderBlock class to create an encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Constructor that creates a decoder block for a transformer
        Arguments:
          - dm is an integer representing the dimensionality of the model
          - h is an integer representing the number of heads
          - hidden is the number of hidden units in the fully connected layer
          - drop_rate is the dropout rate
        Public instance attributes:
          - mha1: the first MultiHeadAttention layer
          - mha2: the second MultiHeadAttention layer
          - dense_hidden: the hidden dense layer with hidden units and relu
          - dense_output: the output dense layer with dm units
          - layernorm1: the first layer norm layer, with epsilon=1e-6
          - layernorm2: the second layer norm layer, with epsilon=1e-6
          - layernorm3: the third layer norm layer, with epsilon=1e-6
          - dropout1: the first dropout layer
          - dropout2: the second dropout layer
          - dropout3: the third dropout layer
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method call that builds a decoder block for a transformer
        Arguments:
          - x is a tensor of shape (batch, target_seq_len, dm) containing
              the input to the decoder block
          - encoder_output is a tensor of shape (batch, input_seq_len, dm)
            containing the output of the encoder
          - training is a boolean to determine if the model is training
          - look_ahead_mask is the mask to be applied to the first multi
            head attention layer
          - padding_mask is the mask to be applied to the second multi head
            attention layer
        Returns:
          - A tensor of shape (batch, target_seq_len, dm) containing the blocks
            output
        """
        # 1st MultiHeadAttention
        mha1, _ = self.mha1(x, x, x, look_ahead_mask)
        mha1 = self.dropout1(mha1, training=training)
        out1 = self.layernorm1(mha1 + x)
        # 2nd MultiHeadAttention
        mha2, _ = self.mha2(out1, encoder_output,
                            encoder_output, padding_mask)
        mha2 = self.dropout2(mha2, training=training)
        out2 = self.layernorm2(mha2 + out1)
        # Dense hidden
        dense_hidden = self.dense_hidden(out2)
        # Dense output
        dense_output = self.dense_output(dense_hidden)
        dense_output = self.dropout3(dense_output, training=training)
        # Output
        output = self.layernorm3(dense_output + out2)
        return output
