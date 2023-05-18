# import torch
# import torch.nn.functional as F
# import torch.nn as nn

# from torch import nn
# from torch.nn import tf.keras.layers.Conv1D

import tensorflow as tf
# import tensorflow_probability as tfp
# def init_weights(m, mean=0.0, std=0.01):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.kernel_initializer.mean = mean
#         m.kernel_initializer.stddev = std


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class AMPBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock, self).__init__()
        self.convs1 = [
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[0],
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros'
                               ),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[1],
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros'),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=dilation[2],
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros')
        ]
        # for conv in self.convs1:
        #     init_weights(conv)
        #self.convs1=map(self.convs1,init_weights)

        self.convs2 = [
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros'),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D( channels, kernel_size, 1, dilation_rate=1,
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros'),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv1D(channels, kernel_size, 1, dilation_rate=1,
                               padding='causal',
                               kernel_initializer='random_normal',
                                bias_initializer='zeros')
        ]
        # for conv in self.convs2:
        #     init_weights(conv)
       # self.convs2.apply(init_weights)

    def call(self, x,training=False):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = tf.keras.layers.LeakyReLU(0.1)(x)
            xt = c1(xt,training=training)
            xt = tf.keras.layers.LeakyReLU(0.1)(x)
            xt = c2(xt,training=training)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs1:
    #         remove_weight_norm(l)
    #     for l in self.convs2:
    #         remove_weight_norm(l)