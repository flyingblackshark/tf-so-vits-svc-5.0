import tensorflow as tf

from vits import commons
from vits.losses import kl_loss
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
import tensorflow_addons as tfa

def patch_batch(batch):
    # Right zero-pad all one-hot text sequences to max input length
    # mel: [freq, length]
    # wav: [1, length]
    # ppg: [len, 1024]
    # pit: [len]
    # spk: [256]
   #data = data.get_single_element()
    _, ids_sorted_decreasing = tf.sort(
        tf.cast([x[0].shape[1] for x in batch],dtype=tf.float32), dim=0, descending=True
    )

    max_spe_len = max([x[0].shape[1] for x in batch])
    max_wav_len = max([x[1].shape[1] for x in batch])
    spe_lengths = tf.zeros(len(batch),dtype=tf.float64)
    wav_lengths =  tf.zeros(len(batch),dtype=tf.float64)
    spe_padded = tf.zeros(
        (len(batch), batch[0][0].shape[0], max_spe_len),dtype=tf.float32)
    wav_padded = tf.zeros((len(batch), 1, max_wav_len),dtype=tf.float32)

    max_ppg_len = max([x[2].shape[0] for x in batch])
    ppg_lengths =tf.zeros(len(batch))
    ppg_padded = tf.zeros((len(batch), max_ppg_len, batch[0][2].shape[1]))
    pit_padded = tf.zeros((len(batch), max_ppg_len))
    spk = tf.zeros((len(batch), batch[0][4].shape[0]))

    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        spe = row[0]
        spe_padded[i, :, : spe.shape[1]] = spe
        spe_lengths[i] = spe.shape[1]

        wav = row[1]
        wav_padded[i, :, : wav.shape[1]] = wav
        wav_lengths[i] = wav.shape[1]

        ppg = row[2]
        ppg_padded[i, : ppg.shape[0], :] = ppg
        ppg_lengths[i] = ppg.shape[0]

        pit = row[3]
        pit_padded[i, : pit.shape[0]] = pit

        spk[i] = row[4]
    # print(ppg_padded.shape)
    # print(ppg_lengths.shape)
    # print(pit_padded.shape)
    # print(spk.shape)
    # print(spe_padded.shape)
    # print(spe_lengths.shape)
    # print(wav_padded.shape)
    # print(wav_lengths.shape)
    return (
        ppg_padded,
        ppg_lengths,
        pit_padded,
        spk,
        spe_padded,
        spe_lengths,
        wav_padded,
        wav_lengths,
    )



class ConditionalGAN(tf.keras.Model):
    def __init__(self, discriminator, generator,hp):#, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
    #    self.latent_dim = latent_dim
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.hp=hp

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        #optim_g.zero_grad()
        
        #test = patch_batch(test)
       # data = _normalize_function
        # data = data.map(_parse_function)
        # data = data.map(_normalize_function)
        # ppg, pit, spec, spk, ppg_l, spec_l = patch_batch(data)
        stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
        stft = TacotronSTFT(filter_length=hp.data.filter_length,
                        hop_length=hp.data.hop_length,
                        win_length=hp.data.win_length,
                        n_mel_channels=hp.data.mel_channels,
                        sampling_rate=hp.data.sampling_rate,
                        mel_fmin=hp.data.mel_fmin,
                        mel_fmax=hp.data.mel_fmax,
                        )
        hp=self.hp
        fake_audio, ids_slice, z_mask, \
            (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = self.generator(
                ppg, pit, spec, spk, ppg_l, spec_l)


        audio = commons.slice_segments(
            audio, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice

        # Mel Loss
        with tf.GradientTape as tape:
            mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
            mel_real = stft.mel_spectrogram(audio.squeeze(1))
            mel_loss = tf.keras.losses.mean_squared_error(mel_fake, mel_real) * hp.train.c_mel

            # Multi-Resolution STFT Loss
            sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
            stft_loss = (sc_loss + mag_loss) * hp.train.c_stft

            # Generator Loss
            res_fake, period_fake, dis_fake = self.discriminator(fake_audio)
            score_loss = 0.0
            for (_, score_fake) in res_fake + period_fake + dis_fake:
                score_loss += tf.mean(tf.pow(score_fake - 1.0, 2))
            score_loss = score_loss / len(res_fake + period_fake + dis_fake)

            # Feature Loss
            res_real, period_real, dis_real = self.discriminator(audio)
            feat_loss = 0.0
            for (feat_fake, _), (feat_real, _) in zip(res_fake + period_fake + dis_fake, res_real + period_real + dis_real):
                for fake, real in zip(feat_fake, feat_real):
                    feat_loss += tf.mean(tf.abs(fake - real))
            feat_loss = feat_loss / len(res_fake + period_fake + dis_fake)
            feat_loss = feat_loss * 2

            # Kl Loss
            loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
            loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

            # Loss
            loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f
        grads = tape.gradient(loss_g,self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads,self.generator.trainble_weights)
        )
        #loss_g.backward()
        #clip_grad_value_(model_g.parameters(),  None)
        #optim_g.step()

        # discriminator
        #optim_d.zero_grad()
        with tf.GradientTape as tape:
            res_fake, period_fake, dis_fake = self.discriminator(fake_audio.detach())
            res_real, period_real, dis_real = self.discriminator(audio)

            loss_d = 0.0
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake + dis_fake, res_real + period_real + dis_real):
                loss_d += tf.mean(tf.pow(score_real - 1.0, 2))
                loss_d += tf.mean(tf.pow(score_fake, 2))
            loss_d = loss_d / len(res_fake + period_fake + dis_fake)
        grads = tape.gradient(loss_d,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads,self.discriminator.trainble_weights)
        )
            #loss_d.backward()
            #clip_grad_value_(model_d.parameters(),  None)
            #optim_d.step()