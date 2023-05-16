# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import tensorflow as tf
import tensorflow_addons as tfa
class DiscriminatorP(tf.keras.Model):
    def __init__(self, hp, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period

        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride
       # tfa.layers.WeightNormalization = weight_norm if hp.mpd.use_spectral_norm == False else spectral_norm

        self.convs = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#1, 
            64, (kernel_size, 1), (stride, 1), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#64, 
            128, (kernel_size, 1), (stride, 1), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#128,
             256, (kernel_size, 1), (stride, 1), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#256,
             512, (kernel_size, 1), (stride, 1), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#512,
             1024, (kernel_size, 1), 1, padding='same')),
        ]
        self.conv_post = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#1024, 
            1,(3, 1), 1, padding='same'))

    def call(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = tf.pad(x, [(0,0),(0,0),(0, n_pad)], "reflect")
            t = t + n_pad
        #x = x.view(b, c, t // self.period, self.period)
        x = tf.reshape(x, [b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x)
            #x = tf.keras.layers.LeakyReLU(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = tf.reshape(x,[x.shape[0],-1])
        return fmap, x


class MultiPeriodDiscriminator(tf.keras.Model):
    def __init__(self, hp):
        super(MultiPeriodDiscriminator, self).__init__()

        self.discriminators = [DiscriminatorP(hp, period) for period in hp.mpd.periods]
        

    def __call__(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
