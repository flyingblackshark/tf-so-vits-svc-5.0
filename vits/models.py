
#import torch

# from torch import nn
# from torch.nn import functional as F
from vits import attentions
from vits import commons
#from vits.modules_grl import SpeakerClassifier
#from vits import modules
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator
import tensorflow as tf
import vits.modules

class TextEncoder(tf.keras.layers.Layer):
    def __init__(self,
              #   in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = tf.keras.layers.Conv1D(filters=hidden_channels, kernel_size=5, padding='same')
        self.pit = tf.keras.layers.Embedding(256, hidden_channels)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = tf.keras.layers.Conv1D(out_channels * 2, 1)

    def call(self, x, x_lengths, f0, training=False):
        x_mask = tf.expand_dims(tf.sequence_mask(x_lengths, x.shape[1],dtype=tf.float32),1)
        x_mask = tf.expand_dims(x_mask,0)
        x = self.pre(x,training=training) * x_mask
        x = x + self.pit(f0,training=training)
        x = self.enc(x * x_mask, x_mask,training=training)
        stats = self.proj(x,training=training) * x_mask
        m, logs = tf.split(stats,2,axis=2)
        z = (m + tf.random.stateless_normal(shape=[-1,m.shape[1],m.shape[2]],seed=[1,2]) * tf.exp(logs)) * x_mask
        return z, m, logs, x_mask,x


class ResidualCouplingBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = []#nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                vits.modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(vits.modules.Flip())

    def call(self, x, x_mask, g=None, reverse=False,training=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse,training=training)
                total_logdet += tf.cast(log_det,tf.float32)
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse,training=training)
                total_logdet += tf.cast(log_det,tf.float32)
            return x, total_logdet

    # def remove_weight_norm(self):
    #     for i in range(self.n_flows):
    #         self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
    #    in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = tf.keras.layers.Conv1D(filters=hidden_channels, kernel_size=1)
        self.enc = vits.modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = tf.keras.layers.Conv1D(
            out_channels * 2, 1)

    def call(self, x, x_lengths, g=None,training=False):
        x_mask = tf.expand_dims(tf.sequence_mask(x_lengths, x.shape[1],dtype=tf.float32), 1)
        x_mask =tf.expand_dims(x_mask,0)
        x = self.pre(x,training=training) * x_mask
        x = self.enc(x, x_mask, g=g,training=training)
        stats = self.proj(x,training=training) * x_mask
        m, logs = tf.split(stats,[self.out_channels,self.out_channels],axis=2)
        z = (m + tf.random.stateless_normal([-1,m.shape[1],m.shape[2]],seed=[1,2]) * tf.exp(logs)) * x_mask
        return z, m, logs, x_mask

    # def remove_weight_norm(self):
    #     self.enc.remove_weight_norm()


class SynthesizerTrn(tf.keras.Model):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = tf.keras.layers.Dense(hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        # self.speaker_classifier = SpeakerClassifier(
        #     hp.vits.hidden_channels,
        #     hp.vits.spk_dim,
        # )
        self.enc_q = PosteriorEncoder(
            #spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def call(self, ppg, pit, spec, spk, ppg_l, spec_l, training=False):
        #ppg = ppg + tf.random.normal(ppg.shape,dtype=tf.float32) * 0.0001
        g = tf.expand_dims(self.emb_g(spk,training=training),0)
            #tf.linalg.normalize(spk,axis=-1),training=training),0)
        z_p, m_p, logs_p, ppg_mask,x = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit),training=training)
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g,training=training)

        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_q, pit, spec_l, self.segment_size)
        audio = self.dec(spk, z_slice, pit_slice,training=training)

        # SNAC to flow
        z_f, logdet_f = self.flow(z_q, spec_mask, g=spk,training=training)
        z_r, logdet_r = self.flow(z_p, spec_mask, g=spk, reverse=True,training=training)

        #spk_preds = self.speaker_classifier(x)
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r)

    def infer(self, ppg, pit, spk, ppg_l):
        z_p, m_p, logs_p, ppg_mask = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit)
        return o


class SynthesizerInfer(tf.keras.Model):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
        #    hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    # def remove_weight_norm(self):
    #     self.flow.remove_weight_norm()
    #     self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, pit, spk, ppg_l, source):
        z_p, m_p, logs_p, ppg_mask = self.enc_p(
            ppg, ppg_l, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
