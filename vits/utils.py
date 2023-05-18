import os
import argparse
import numpy as np

import tensorflow as tf
from scipy.io.wavfile import read
from omegaconf import OmegaConf

MATPLOTLIB_FLAG = False


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return tf.convert_to_tensor(data.astype(np.float32)), sampling_rate


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    #  is_torch = isinstance(f0, tf.Tensor)
    f0_mel = 1127 * tf.math.log((1 + f0 / 700)) 
    # if is_torch else 1127 * \
    #     np.log(1 + f0 / 700)
    
    f0_mel=tf.where(f0_mel > 0, (f0_mel - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1,f0_mel)

    f0_mel=tf.where(f0_mel <= 1,1,f0_mel)
    f0_mel=tf.where(f0_mel > f0_bin - 1,f0_bin - 1,f0_mel)
    f0_coarse = tf.cast((f0_mel + 0.5),tf.int64) #if is_torch else np.rint(f0_mel).astype(np.int)
    assert tf.reduce_max(f0_coarse) <= 255 and tf.reduce_min(f0_coarse) >= 1, (tf.reduce_max(f0_coarse), tf.reduce_min(f0_coarse))
    return f0_coarse


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.yaml",
                        help='YAML file for configuration')
    args = parser.parse_args()
    hparams = OmegaConf.load(args.config)
    model_dir = os.path.join("./logs", hparams.train.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_save_path = os.path.join(model_dir, "config.json")
    os.system(f"cp {args.config} {config_save_path}")
    hparams.model_dir = model_dir
    return hparams
