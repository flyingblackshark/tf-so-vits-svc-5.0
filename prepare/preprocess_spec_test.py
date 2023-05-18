import os
import tensorflow as tf
import argparse
#import tensorflow_io as tfio
from vits import spectrogram
from vits import utils
from omegaconf import OmegaConf
def complex_to_float(complex_num):
    real = tf.cast(tf.math.real(complex_num),dtype=tf.bfloat16)
    imag = tf.cast(tf.math.imag(complex_num),dtype=tf.bfloat16)
    return tf.sqrt(real**2+imag**2+1e-6)
def spectrogram_tf(y, n_fft, sampling_rate, hop_size, win_size):
    if tf.reduce_min(y) < -1.0:
        print("min value is ", tf.reduce_min(y))
    if tf.reduce_max(y) > 1.0:
        print("max value is ", tf.reduce_max(y))

    # global hann_window
    # dtype_device = str(y.dtype) + "_" + str(y.device)
    # wnsize_dtype_device = str(win_size) + "_" + dtype_device
    # if wnsize_dtype_device not in hann_window:
    #     hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
    #         dtype=y.dtype, device=y.device
    #     )
    temp = tf.expand_dims(y,1)
    y = tf.pad(
        temp,
        [(0,0),(0,0),(int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2))],
        mode="reflect",
    )
    y = tf.squeeze(y,1)

    spec = tf.signal.stft(
        signals = y,
        fft_length=n_fft,
        frame_step=hop_size,
        frame_length=win_size,
        #window=hann_window[wnsize_dtype_device],
        window_fn=tf.signal.hann_window,
        pad_end=False
       # center=center,
        #pad_mode="reflect",
       # normalized=False,
       # onesided=True,
       # return_complex=False,
    )

    #spec = tf.sqrt(tf.reduce_sum(tf.pow(spec,2),axis=-1) + 1e-6)
    new_spec = tf.map_fn(complex_to_float,spec,dtype=tf.bfloat16)
    return new_spec
def compute_spec(hps, filename, specname):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    assert sampling_rate == hps.sampling_rate, f"{sampling_rate} is not {hps.sampling_rate}"
    audio_norm = audio / hps.max_wav_value
    audio_norm = tf.expand_dims(audio_norm,0)
    n_fft = hps.filter_length
    sampling_rate = hps.sampling_rate
    hop_size = hps.hop_length
    win_size = hps.win_length
    # spec = spectrogram.spectrogram_torch(
    #     audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False)
    #spec = tfio.audio.spectrogram(audio_norm,n_fft,win_size,hop_size)
    spec = spectrogram_tf(audio_norm, n_fft, sampling_rate, hop_size, win_size)
    spec = tf.squeeze(spec, 0)
    spec = tf.io.serialize_tensor(spec)
    tf.io.write_file(specname,spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-s", "--spe", help="spe", dest="spe")
    args = parser.parse_args()
    print(args.wav)
    print(args.spe)
   # os.makedirs(args.spe)
    wavPath = args.wav
    spePath = args.spe
    hps = OmegaConf.load("./configs/base.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
           # os.makedirs(f"./{spePath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in os.listdir(f"./{wavPath}/{spks}"):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    compute_spec(hps.data, f"{wavPath}/{spks}/{file}.wav", f"{spePath}/{spks}/{file}.pt")
        else:
            file = spks
            if file.endswith(".wav"):
                # print(file)
                file = file[:-4]
                compute_spec(hps.data, f"{wavPath}/{file}.wav", f"{spePath}/{file}.pt")





