import tensorflow as tf
from vits import commons
from vits.losses import kl_loss
from vits.models import SynthesizerTrn
from vits_decoder.discriminator import Discriminator
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import STFTLoss

def read_tfrecord(example):
    feature=({
        "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "spk": tf.io.FixedLenFeature([], tf.string, default_value=''),
    })
    example = tf.io.parse_single_example(example, feature)
    spe = tf.cast(tf.io.parse_tensor(example["spe"],out_type=tf.float32),dtype=tf.bfloat16)
    wav = tf.cast(tf.io.parse_tensor(example["wav"],out_type=tf.float32),dtype=tf.bfloat16)
    ppg = tf.cast(tf.io.parse_tensor(example["ppg"],out_type=tf.float32),dtype=tf.bfloat16)
    pit = tf.cast(tf.io.parse_tensor(example["pit"],out_type=tf.float64),dtype=tf.bfloat16)
    spk = tf.cast(tf.io.parse_tensor(example["spk"],out_type=tf.float32),dtype=tf.bfloat16)
    return spe,wav,ppg,pit,spk
def load_dataset():
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        "test.tfrecords"
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset
    
    
   
class MultiResolutionSTFTLoss(tf.keras.layers.Layer):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 device,
                 resolutions,
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            resolutions (list): List of (FFT size, hop size, window length).
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.stft_losses = []
        for fs, ss, wl in resolutions:
            self.stft_losses += [STFTLoss(device, fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
def train(rank, args, chkpt_path, hp, hp_str):
    #try TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.set_global_policy(policy)
    strategy = tf.distribute.TPUStrategy(resolver)
    tf.random.set_seed(hp.train.seed)
    dataset = load_dataset()
    dataset = dataset.shuffle(2)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(1)
    model_g = SynthesizerTrn(
    hp.data.filter_length // 2 + 1,
    hp.data.segment_size // hp.data.hop_length,
    hp)
        
    model_d=Discriminator(hp)
    d_optimizer=tf.keras.optimizers.AdamW(learning_rate=hp.train.learning_rate, beta_1=hp.train.betas[0],beta_2=hp.train.betas[1], epsilon=hp.train.eps)
    g_optimizer=tf.keras.optimizers.AdamW(learning_rate=hp.train.learning_rate, beta_1=hp.train.betas[0],beta_2=hp.train.betas[1], epsilon=hp.train.eps)
    stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
    
    stft = TacotronSTFT(filter_length=hp.data.filter_length,
                        hop_length=hp.data.hop_length,
                        win_length=hp.data.win_length,
                        n_mel_channels=hp.data.mel_channels,
                        sampling_rate=hp.data.sampling_rate,
                        mel_fmin=hp.data.mel_fmin,
                        mel_fmax=hp.data.mel_fmax,
                        center=False)


    vpr_loss = tf.keras.losses.CosineSimilarity()
    epochs = 200
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (spec,wav,ppg,pit,spk) in enumerate(dataset):
            with tf.device('/TPU:0'):
                with tf.GradientTape(persistent= True) as tape:
                    audio = tf.squeeze(wav,0)
                    len_pit = pit.shape[1]
                    len_ppg = ppg.shape[1]
                    len_min = min(len_pit,len_ppg)  
                    pit = pit[:len_min]
                    ppg = ppg[:len_min, :]
                    spec = spec[:,:len_min,: ]        
                    ppg_l = ppg.shape[1]
                    spec_l =spec.shape[1]
            
                    fake_audio, ids_slice, z_mask, \
                        (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = model_g(
                            ppg, pit, spec, spk, ppg_l, spec_l,training=True)
                    audio = commons.slice_segments(
                        audio, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
                    #spk_loss = vpr_loss(spk, spk_preds, tf.cast(spk_preds.size(0),tf.bfloat16).fill_(1.0))

                    mel_fake = stft.mel_spectrogram(tf.expand_dims(fake_audio,1))
                    mel_real = stft.mel_spectrogram(tf.expand_dims(audio,1))

                    mel_loss = tf.keras.losses.MAE(mel_fake, mel_real) * hp.train.c_mel
                    sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
                    stft_loss = (sc_loss+mag_loss) * hp.train.c_stft
                    disc_fake = model_d(fake_audio)
                    score_loss = 0.0
                    for (_, score_fake) in disc_fake:
                        score_loss += tf.reduce_mean(tf.pow(score_fake - 1.0, 2))
                    score_loss = score_loss / len(disc_fake)


                    disc_real = model_d(audio)
                    feat_loss = 0.0
                    for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                        for fake, real in zip(feat_fake, feat_real):
                            feat_loss += tf.reduce_mean(tf.abs(fake - real))
                    feat_loss = feat_loss / len(disc_fake)
                    feat_loss = feat_loss * 2
            
                    # Loss
                    loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
                    loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl

                    loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 
                d_gradients = tape.gradient(loss_d, model_d.trainable_variables)
                d_optimizer.apply_gradients(zip(d_gradients, model_d.trainable_weights),skip_aggregate_gradients=True)
                with tf.GradientTape(persistent= True) as tape:
                    disc_fake = model_d(fake_audio.detach())
                    disc_real = model_d(audio)

                    loss_d = 0.0
                    for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                        loss_d += tf.reduce_mean(tf.pow(score_real - 1.0, 2))
                        loss_d += tf.reduce_mean(tf.pow(score_fake, 2))
                    loss_d = loss_d / len(disc_fake)
                g_gradients = tape.gradient(loss_g, model_g.trainable_variables)
                g_optimizer.apply_gradients(zip(g_gradients, model_g.trainable_weights),skip_aggregate_gradients=True)
                loss_g = loss_g
                loss_d = loss_d
                loss_s = stft_loss
                loss_m = mel_loss
                loss_k = loss_kl_f
                loss_r = loss_kl_r
            
            print("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f | step %d" % (
                loss_g, loss_m, loss_s, loss_d, loss_k, loss_r,step))
                
