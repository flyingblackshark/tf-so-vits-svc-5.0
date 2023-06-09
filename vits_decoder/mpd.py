# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import tensorflow as tf
#import tensorflow_probability as tfp
class DiscriminatorP(tf.keras.layers.Layer):
    def __init__(self, hp, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period

        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride
       # tfa.layers.WeightNormalization = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = [
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#1, 
            64, (kernel_size, 1), (stride, 1), padding='same'),
            #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#64, 
            128, (kernel_size, 1), (stride, 1), padding='same'),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#128,
             256, (kernel_size, 1), (stride, 1), padding='same'),
           #tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#256,
             512, (kernel_size, 1), (stride, 1), padding='same'),
          # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#512,
             1024, (kernel_size, 1), 1, padding='same'),
        ]
        self.conv_post = tf.keras.layers.Conv2D( 
            1,(3, 1), 1, padding='same')
        #tfa.layers.WeightNormalization(
        

    def call(self, x,training=False):
        fmap = []

        # 1d to 2d
        b, t, c = x.shape
        #b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = tf.pad(x, [(0,0),(0, n_pad),(0,0)], "reflect")
            t = t + n_pad
        #x = x.view(b, c, t // self.period, self.period)
        x = tf.reshape(x, [b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x,training=training)
            x = tf.keras.layers.LeakyReLU(self.LRELU_SLOPE)(x)
            fmap.append(x)
        x = self.conv_post(x,training=training)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = tf.reshape(x,[x.shape[0],-1])
        return fmap, x


class MultiPeriodDiscriminator(tf.keras.layers.Layer):
    def __init__(self, hp):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = [DiscriminatorP(hp, period) for period in hp.mpd.periods]
        

    def call(self, x,training=False):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,training=training))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
