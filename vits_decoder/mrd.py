# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import functools
import tensorflow as tf
import tensorflow_addons as tfa
class DiscriminatorR(tf.keras.layers.Layer):
    def __init__(self, hp, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        #tfa.layers.WeightNormalization = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = [
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#1, 
                                                                  filters=32, kernel_size=(3, 9), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same')),
            tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 3), padding='same')),
        ]
        self.conv_post = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(#32,
                                                                                1, (3, 3), padding='same'))

    def call(self, x):
        fmap = []
        
        x = self.spectrogram(x)
        x = tf.expand_dims(x,1)
      #  x = tf.expand_dims(x,1)
        for l in self.convs:
            x = l(x)
            #x = tf.keras.layers.LeakyReLU(self.LRELU_SLOPE)(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = tf.reshape(x,[x.shape[0],-1])
        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = tf.pad(x, [(0,0),(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2))], mode='reflect')
        x = tf.squeeze(x,1)
       # x = tf.signal.stft(signals=x, frame_length=n_fft, frame_step=hop_length, fft_length=win_length,pad_end=False,window_fn=functools.partial(tf.signal.hann_window, periodic=True)) #[B, F, TT, 2]
        x = tf.signal.stft(signals=x, frame_length=win_length, frame_step=hop_length, fft_length=n_fft)
        mag = tf.cast(x,dtype=tf.float32)
       # mag = tf.norm(x, ord=2, axis =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(tf.keras.layers.Layer):
    def __init__(self, hp):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        

    def call(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
