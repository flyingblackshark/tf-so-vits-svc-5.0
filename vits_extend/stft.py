# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import random
# import torch
# import torch.utils.data
import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
#import librosa
#import tensorflow_io as tfio
from librosa.filters import mel as librosa_mel_fn

# def mel_filterbank(sr, n_mels, n_fft, fmin, fmax):
#   """Computes the mel filterbank.

#   Args:
#     sr: The sample rate of the input signal.
#     n_mels: The number of mel bands.
#     n_fft: The number of samples in the FFT.
#     fmin: The minimum frequency of the mel bands.
#     fmax: The maximum frequency of the mel bands.

#   Returns:
#     The mel filterbank, a tensor of shape [n_mels, n_fft // 2 + 1].
#   """

#   # Compute the mel frequencies.
#   #mel_frequencies = librosa.mel_frequencies(sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)

#   # Compute the mel filterbank.
#   mel_filterbank = librosa.filters.mel(sr=sr, n_mels=n_mels, n_fft=n_fft, fmin=fmin, fmax=fmax)

#   return mel_filterbank
class TacotronSTFT():
    def __init__(self, filter_length=512, hop_length=160, win_length=512,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=None, center=False):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)

        self.mel_basis = tf.cast(tf.transpose(tf.convert_to_tensor(mel), perm=[1,0]),dtype=tf.bfloat16)
        # mel = librosa_mel_fn(
        #     sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
            # Register mel_basis buffer
        # self.mel_basis = tf.Variable(
        #     mel_filterbank(sr=sampling_rate, n_mels=n_mel_channels, n_fft=filter_length, fmin=mel_fmin, fmax=mel_fmax),
        #     trainable=False,
        # )

        # Register hann_window buffer
        # self.hann_window = tf.Variable(
        #     tf.signal.hann_window(win_length), trainable=False
        # )
        # mel_basis = tf.convert_to_tensor(mel)#.float().to(device)
        # hann_window = tf.signal.hann_window(win_length)#.to(device)

        # self.register_buffer('mel_basis', mel_basis)
        # self.register_buffer('hann_window', hann_window)

    # def linear_spectrogram(self, y):
    #     assert (tf.math.min(y.data) >= -1)
    #     assert (tf.math.max(y.data) <= 1)

    #     y = tf.pad(y.unsqueeze(1),
    #                                 (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)),
    #                                 mode='reflect')
    #     y = y.squeeze(1)
    #     spec = tf.signal.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
    #                       center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    #     spec = tf.norm(spec, p=2, dim=-1)

    #     return spec
    
        
    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(tf.reduce_min(y) >= -1)
        assert(tf.reduce_max(y) <= 1)
        paddings =  tf.constant([(0,0),(0,0),
        (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2))
        ])
        y = tf.squeeze(y,-1)
        y = tf.pad(tensor=y,paddings=paddings,
            mode='CONSTANT')
        y = tf.squeeze(y,1)
        #y = tf.squeeze(y,0)
        spec =tf.signal.stft(
        signals = tf.cast(y,dtype=tf.float32),
        fft_length=self.n_fft,
        frame_step=self.hop_size,
        frame_length=self.win_size,
        window_fn=tf.signal.hann_window,
        pad_end=False
        )
        def complex_to_float(complex_num):
            real = tf.cast(tf.math.real(complex_num),dtype=tf.bfloat16)
            imag = tf.cast(tf.math.imag(complex_num),dtype=tf.bfloat16)
            return tf.sqrt(real**2+imag**2)+ 1e-9
        spec = tf.map_fn(complex_to_float,spec,dtype=tf.bfloat16)
        spec = tf.matmul(spec,self.mel_basis )
        spec = self.spectral_normalize_torch(spec)
        spec = tf.cast(spec,dtype=tf.bfloat16)
        return spec

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return tf.math.log(tf.clip_by_value(x, clip_value_min=clip_val,clip_value_max=tf.bfloat16.max) * C)
