# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from omegaconf import OmegaConf
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
import tensorflow as tf
#import tensorflow_probability as tfp

class Discriminator(tf.keras.Model):
    def __init__(self, hp):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator(hp)
        self.MPD = MultiPeriodDiscriminator(hp)


    def call(self, x):
        r = self.MRD(x)
        p = self.MPD(x)

        return r + p


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

