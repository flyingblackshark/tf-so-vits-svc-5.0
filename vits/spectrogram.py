# import torch
# import torch.utils.data
import tensorflow as tf
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return tf.log(tf.clip_by_value(x, clip_value_min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return tf.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

# @tf.function
# def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
#     if tf.keras.backend.min(y) < -1.0:
#         print("min value is ", tf.keras.backend.min(y))
#     if tf.keras.backend.max(y) > 1.0:
#         print("max value is ", tf.keras.backend.max(y))

#     global hann_window
#     dtype_device = str(y.dtype) + "_" + str(y.device)
#     wnsize_dtype_device = str(win_size) + "_" + dtype_device
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = tf.signal.hann_window(win_size)
#     y = tf.expand_dims(y,1)
    
#     y = tf.pad(
#         y,#y.unsqueeze(1),
#         (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#         mode="reflect",
#     )
#     y = tf.squeeze(y,1)

#     spec = tf.signal.stft(
#         y,
#         n_fft,
#         hop_length=hop_size,
#         win_length=win_size,
#         window=hann_window[wnsize_dtype_device],
#         center=center,
#         pad_mode="reflect",
#         normalized=False,
#         onesided=True,
#         return_complex=False,
#     )

#     spec = tf.sqrt(spec.pow(2).sum(-1) + 1e-6)
#     return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = tf.convert_to_tensor(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    spec = tf.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


# def mel_spectrogram_torch(
#     y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
# ):
#     if tf.math.min(y) < -1.0:
#         print("min value is ", tf.math.min(y))
#     if tf.math.max(y) > 1.0:
#         print("max value is ", tf.math.max(y))

#     global mel_basis, hann_window
#     dtype_device = str(y.dtype) + "_" + str(y.device)
#     fmax_dtype_device = str(fmax) + "_" + dtype_device
#     wnsize_dtype_device = str(win_size) + "_" + dtype_device
#     if fmax_dtype_device not in mel_basis:
#         mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis[fmax_dtype_device] = tf.convert_to_tensor(mel).to(
#             dtype=y.dtype, device=y.device
#         )
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = tf.signal.hann_window(win_size).to(
#             dtype=y.dtype, device=y.device
#         )

#     y = tf.pad(
#         y.unsqueeze(1),
#         (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#         mode="reflect",
#     )
#     y = y.squeeze(1)

#     spec = tf.signal.stft(
#         y,
#         n_fft,
#         hop_length=hop_size,
#         win_length=win_size,
#         window=hann_window[wnsize_dtype_device],
#         center=center,
#         pad_mode="reflect",
#         normalized=False,
#         onesided=True,
#         return_complex=False,
#     )

#     spec = tf.sqrt(spec.pow(2).sum(-1) + 1e-6)

#     spec = tf.matmul(mel_basis[fmax_dtype_device], spec)
#     spec = spectral_normalize_torch(spec)

#     return spec
