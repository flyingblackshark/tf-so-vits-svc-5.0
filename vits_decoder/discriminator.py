# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from omegaconf import OmegaConf
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
import tensorflow as tf
#import tensorflow_probability as tfp

class DiscriminatorS(tf.keras.layers.Layer):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.convs = [
            tf.keras.layers.Conv1D(16, 15, 1, padding='same'),
            tf.keras.layers.Conv1D(64, 41, 4,padding='same'),
            tf.keras.layers.Conv1D(256, 41, 4,  padding='same'),
            tf.keras.layers.Conv1D(1024, 41, 4, padding='same'),
            tf.keras.layers.Conv1D(1024, 41, 4, padding='same'),
            tf.keras.layers.Conv1D(1024, 5, 1, padding='same')
        ]
        self.conv_post = tf.keras.layers.Conv1D(1, 3, 1, padding='same')
        

    def call(self, x,training=False):
        fmap = []
        for l in self.convs:
            x = l(x,training=training)
            x = tf.keras.layers.LeakyReLU(0.1)(x)
            fmap.append(x)
        x = self.conv_post(x,training=training)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = tf.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]


class Discriminator(tf.keras.Model):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)
        self.DIS = DiscriminatorS()

    def call(self, x):
        return self.MRD(x), self.MPD(x), self.DIS(x)


if __name__ == '__main__':
    hp = OmegaConf.load('../config/default.yaml')
    model = Discriminator(hp)

    x = tf.random.normal(3, 1, 16384)
    print(x.shape)

    mrd_output, mpd_output = model(x)
    for features, score in mpd_output:
        for feat in features:
            print(feat.shape)
        print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

