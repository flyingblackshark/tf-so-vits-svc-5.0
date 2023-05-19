# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import functools
import tensorflow as tf
#import tensorflow_probability as tfp
class DiscriminatorR(tf.keras.layers.Layer):
    def __init__(self, hp, resolution):
        super(DiscriminatorR, self).__init__()

        self.resolution = resolution
        self.LRELU_SLOPE = hp.mpd.lReLU_slope

        #tfa.layers.WeightNormalization = weight_norm if hp.mrd.use_spectral_norm == False else spectral_norm

        self.convs = [
           # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#1, 
                                                                  filters=32, kernel_size=(3, 9), padding='same'),
          # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same'),
          # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same'),
          # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 9), strides=(1, 2), padding='same'),
         # tfp.layers.weight_norm.WeightNorm(
            tf.keras.layers.Conv2D(#32, 
                                                                  filters=32, kernel_size=(3, 3), padding='same'),
        ]
        self.conv_post = tf.keras.layers.Conv2D( 1, (3, 3), padding='same')
         #tfa.layers.WeightNormalization(
            

    def call(self, x,training=False):
        fmap = []
        
        x = self.spectrogram(x)
        x = tf.expand_dims(x,1)
      #  x = tf.expand_dims(x,1)
        for l in self.convs:
            x = l(x,training=training)
            x = tf.keras.layers.LeakyReLU(self.LRELU_SLOPE)(x)
            fmap.append(x)
        x = self.conv_post(x,training=training)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = tf.reshape(x,[x.shape[0],-1])
        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = tf.pad(x, [(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),(0,0)], mode='reflect')
        x = tf.squeeze(x,-1)
        x = tf.signal.stft(signals=tf.cast(x,tf.float32), frame_length=win_length, frame_step=hop_length, fft_length=n_fft,window_fn=tf.signal.hann_window)

        mag = tf.map_fn(complex_to_float,x,dtype=tf.float32)#tf.cast(x,dtype=tf.float32)
       # mag = tf.norm(x, ord=2, axis =-1) #[B, F, TT]

        return mag

def complex_to_float(complex_num):
    real = tf.cast(tf.math.real(complex_num),dtype=tf.float32)
    imag = tf.cast(tf.math.imag(complex_num),dtype=tf.float32)
    return tf.sqrt(real**2+imag**2)
class MultiResolutionDiscriminator(tf.keras.layers.Layer):
    def __init__(self, hp):
        super(MultiResolutionDiscriminator, self).__init__()
        self.resolutions = eval(hp.mrd.resolutions)
        self.discriminators = [DiscriminatorR(hp, resolution) for resolution in self.resolutions]
        

    def call(self, x,training=False):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,training=training))

        return ret  # [(feat, score), (feat, score), (feat, score)]
