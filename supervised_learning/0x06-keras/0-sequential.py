
#!/usr/bin/env python3
"""
    module
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ function """
    out = K.Sequential()
    reg = K.regularizers.L2(lambtha)
    for idx, layer in enumerate(layers):
        out.add(
            K.layers.Dense(
                layer,
                activation=activations[idx],
                input_shape=(nx,),
                kernel_regularizer=reg
            )
        )
        if idx < (len(layers) - 1):
            out.add(K.layers.Dropout(1 - keep_prob))
    return out
