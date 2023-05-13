# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import tensorflow as tf
import tensorflow_addons as tfa
class DiscriminatorR(tf.Module):
    def __init__(self, hp, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        #tfa.layers.WeightNormalization = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#1, 
                                                                  32, (3, 9), padding='valid')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  32, (3, 9), strides=(1, 2), padding='valid')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  32, (3, 9), strides=(1, 2), padding='valid')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  32, (3, 9), strides=(1, 2), padding='valid')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  32, (3, 3), padding='valid')),
        ]
        self.conv_post = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32,
                                                                                1, (3, 3), padding='valid'))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = x.reshape([x.shape[0],-1])
        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = tf.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = tf.signal.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=False) #[B, F, TT, 2]
        mag = tf.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(tf.Module):
    def __init__(self, hp):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
