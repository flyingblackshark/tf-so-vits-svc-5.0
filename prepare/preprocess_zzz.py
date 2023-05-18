#from tqdm import tqdm
#from torch.utils.data import DataLoader
from omegaconf import OmegaConf
# from vits.data_utils import TextAudioSpeakerSet
# from vits.data_utils import TextAudioSpeakerCollate
# from vits.data_utils import DistributedBucketSampler
import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
hps = OmegaConf.load("./configs/base.yaml")

def load_filepaths(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths
items = load_filepaths("files/valid.txt")
hps = hps.data


lengths = []
items_new = []
items_min = int(hps.segment_size / hps.hop_length * 4)  # 1 S
items_max = int(hps.segment_size / hps.hop_length * 16)  # 4 S

def read_wav(filename):
        audio = tf.io.read_file(filename)
        audio, sampling_rate = tf.audio.decode_wav(audio)
        assert sampling_rate == hps.sampling_rate, f"error: this sample rate of {filename} is {sampling_rate}"
        audio_norm = audio / hps.max_wav_value
        #audio_norm = audio_norm.unsqueeze(0)
        audio_norm = tf.expand_dims(audio_norm,0)
        return audio_norm

with tf.io.TFRecordWriter("./test.tfrecords") as file_writer:
    for wavpath, spec, pitch, ppg, spk in items:
        if not os.path.isfile(wavpath):
            continue
        if not os.path.isfile(spec):
            continue
        if not os.path.isfile(pitch):
            continue
        if not os.path.isfile(ppg):
            continue
        if not os.path.isfile(spk):
            continue
        temp = np.load(pitch)
        usel = int(temp.shape[0] - 1)  # useful length
        if (usel < items_min):
            continue
        if (usel >= items_max):
            usel = items_max
        items_new.append([wavpath, spec, pitch, ppg, spk, usel])
        lengths.append(usel)
    items = items_new
    #lengths = lengths
    for item in items:
        wav = item[0]
        spe = item[1]
        pit = item[2]
        ppg = item[3]
        spk = item[4]
        use = item[5]
        wav = read_wav(wav)
        spe = tf.io.parse_tensor(tf.io.read_file(spe),out_type=tf.float32)

        pit = np.load(pit)
        ppg = np.load(ppg)
        ppg = np.repeat(ppg, 2, 0)  # 320 PPG -> 160 * 2
        spk = np.load(spk)

        pit = tf.convert_to_tensor(pit)
        ppg = tf.convert_to_tensor(ppg)
        spk = tf.convert_to_tensor(spk)
        len_pit = pit.shape[0]
        len_ppg = ppg.shape[0] - 2 # for safe
        len_min = min(len_pit, len_ppg)
        len_wav = len_min * hps.hop_length

        pit = pit[:len_min]
        ppg = ppg[:len_min, :]

        spe = spe[:len_min, :]
        wav = wav[:, :len_wav]
        if len_min > use:
            max_frame_start = ppg.shape[0] - use - 1
            frame_start = 0#random.randint(0, max_frame_start)
            frame_end = frame_start + use

            pit = pit[frame_start:frame_end]
            ppg = ppg[frame_start:frame_end, :]
            spe = spe[frame_start:frame_end, :]

            wav_start = frame_start * hps.hop_length
            wav_end = frame_end * hps.hop_length
            wav = wav[:, wav_start:wav_end]
        #import tempfile
        #example_path = os.path.join(tempfile.gettempdir(), "example.tfrecords")
        spe = tf.io.serialize_tensor(spe)
        wav = tf.io.serialize_tensor(wav)
        ppg = tf.io.serialize_tensor(ppg)
        pit = tf.io.serialize_tensor(pit)
        spk = tf.io.serialize_tensor(spk)
        
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "spe": tf.train.Feature(bytes_list=tf.train.BytesList(value=[spe.numpy()])),
            "wav": tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav.numpy()])),
            "ppg": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ppg.numpy()])),
            "pit": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pit.numpy()])),
            "spk": tf.train.Feature(bytes_list=tf.train.BytesList(value=[spk.numpy()])),
        })).SerializeToString()
        file_writer.write(record_bytes)
# dataset = TextAudioSpeakerSet("files/valid.txt", hps.data)

# for _ in tqdm(dataset):
#     pass


# sampler = DistributedBucketSampler(
#     dataset,
#     4,
#     [150, 300, 450],
#     num_replicas=1,
#     rank=0,
#     shuffle=True)
# collate_fn = TextAudioSpeakerCollate()
# loader = DataLoader(dataset, num_workers=0, shuffle=False, pin_memory=True,
#                     collate_fn=collate_fn, batch_sampler=sampler)


# for _ in tqdm(loader):
#     pass
