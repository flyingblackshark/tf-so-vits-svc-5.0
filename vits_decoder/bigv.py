# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# from torch import nn
# from torch.nn import tf.keras.layers.Conv1D
# from torch.nn.utils import tfa.layers.WeightNormalization, remove_weight_norm

import tensorflow as tf
import tensorflow_addons as tfa
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AMPBlock(tf.keras.Model):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock, self).__init__()
        self.convs1 = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[0],
                               padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[1],
                               padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[2],
                               padding='same'))
        ]
       # self.convs1.apply(init_weights)

        self.convs2 = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                               padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=1,
                               padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                               padding='same'))
        ]
       # self.convs2.apply(init_weights)

    def call(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = tf.keras.layers.LeakyReLU(0.1)(x)
            xt = c1(xt)
            xt = tf.keras.layers.LeakyReLU(0.1)(x)
            xt = c2(xt)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs1:
    #         remove_weight_norm(l)
    #     for l in self.convs2:
    #         remove_weight_norm(l)