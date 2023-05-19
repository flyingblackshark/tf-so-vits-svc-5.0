# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

# import torch
# import torch.nn.functional as F
import tensorflow as tf
#import tensorflow_addons as tfa
import functools
def stft(x, fft_size, hop_size, win_length):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x =tf.squeeze(x,-1)
    x_stft = tf.signal.stft(signals=tf.cast(x,dtype=tf.float32), frame_length=win_length, frame_step=hop_size,fft_length=fft_size,window_fn=tf.signal.hann_window)
    real = tf.math.real(x_stft)#x_stft[..., 0]
    imag = tf.math.imag(x_stft)#x_stft[..., 1]
    real=tf.cast(real,tf.bfloat16)
    imag=tf.cast(imag,tf.bfloat16)
    # real=tf.squeeze(real,0)
    # imag=tf.squeeze(imag,0)
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    temp = tf.sqrt(tf.clip_by_value(real ** 2 + imag ** 2, clip_value_min=1e-7,clip_value_max=tf.bfloat16.max))
    temp = tf.cast(temp,tf.bfloat16)
    return tf.transpose(temp,perm=[0,1,2])
    


class SpectralConvergengeLoss(tf.keras.losses.Loss):
    """Spectral convergence loss module."""

    # def __init__(self):
    #     """Initilize spectral convergence loss module."""
    #     super(SpectralConvergengeLoss, self).__init__()

    def call(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return tf.norm(y_mag - x_mag) / tf.norm(y_mag)


class LogSTFTMagnitudeLoss(tf.keras.losses.Loss):
    """Log STFT magnitude loss module."""

    # def __init__(self):
    #     """Initilize los STFT magnitude loss module."""
    #     super(LogSTFTMagnitudeLoss, self).__init__()

    def call(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return tf.keras.losses.mean_absolute_error(tf.math.log(y_mag), tf.math.log(x_mag))


class STFTLoss(tf.keras.losses.Loss):
    """STFT loss module."""

    def __init__(self,fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__(tf.keras.losses.Reduction.SUM)
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergenge_loss = SpectralConvergengeLoss(tf.keras.losses.Reduction.SUM)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(tf.keras.losses.Reduction.SUM)

    def call(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        
        x_mag = stft(x=x, fft_size=self.fft_size, hop_size=self.shift_size, win_length=self.win_length)
        y_mag = stft(x=y, fft_size=self.fft_size, hop_size=self.shift_size, win_length=self.win_length)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss+mag_loss


# class MultiResolutionSTFTLoss(tf.keras.Model):
#     """Multi resolution STFT loss module."""

#     def __init__(self,
#                  resolutions,):
#                #  window="hann_window"):
#         """Initialize Multi resolution STFT loss module.
#         Args:
#             resolutions (list): List of (FFT size, hop size, window length).
#             window (str): Window function type.
#         """
#         super(MultiResolutionSTFTLoss, self).__init__()
#         self.stft_losses = []#torch.nn.ModuleList()
#         for fs, ss, wl in resolutions:
#             #self.stft_losses += [STFTLoss(fs, ss, wl)]
#             self.stft_losses.append(STFTLoss(fs, ss, wl))

#     def call(self, x, y):
#         """Calculate forward propagation.
#         Args:
#             x (Tensor): Predicted signal (B, T).
#             y (Tensor): Groundtruth signal (B, T).
#         Returns:
#             Tensor: Multi resolution spectral convergence loss value.
#             Tensor: Multi resolution log STFT magnitude loss value.
#         """
#         sc_loss = 0.0
#         mag_loss = 0.0
#         for f in self.stft_losses:
#             sc_l, mag_l = f(x, y)
#             sc_loss += sc_l
#             mag_loss += mag_l

#         sc_loss /= len(self.stft_losses)
#         mag_loss /= len(self.stft_losses)

#        # return sc_loss, mag_loss
#         return sc_loss + mag_loss # for test purpose
        
