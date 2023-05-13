# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from omegaconf import OmegaConf
# from torch.nn.utils import tfa.layers.WeightNormalization
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
import tensorflow as tf
import tensorflow_addons as tfa

class DiscriminatorS(tf.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.convs = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#1, 
            16, 15, 1, padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#16,
             64, 41, 4, groups=4, padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#64,
             256, 41, 4, groups=16, padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#256,
             1024, 41, 4, groups=64, padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#1024, 
            1024, 41, 4, groups=256, padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#1024,
             1024, 5, 1, padding='same')),
        ]
        self.conv_post = tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(#1024,
             1, 3, 1, padding='same'))

    def __call__(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            #x = tf.keras.layers.LeakyReLU(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = x.reshape([x.shape[0],-1])
        return [(fmap, x)]


class Discriminator(tf.Module):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)
        self.DIS = DiscriminatorS()

    def __call__(self, x):
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

