import tensorflow as tf
from vits import commons
from vits.losses import kl_loss
from vits.models import SynthesizerTrn
from vits_decoder.discriminator import Discriminator
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import STFTLoss
import tensorflow.experimental.numpy as tnp
def read_tfrecord(example):
    feature=({
        "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "spk": tf.io.FixedLenFeature([], tf.string, default_value=''),
    })
    example = tf.io.parse_single_example(example, feature)
    spe = tf.cast(tf.io.parse_tensor(example["spe"],out_type=tf.float32),dtype=tf.float32)
    wav = tf.cast(tf.io.parse_tensor(example["wav"],out_type=tf.float32),dtype=tf.float32)
    ppg = tf.cast(tf.io.parse_tensor(example["ppg"],out_type=tf.float32),dtype=tf.float32)
    pit = tf.cast(tf.io.parse_tensor(example["pit"],out_type=tf.float64),dtype=tf.float32)
    spk = tf.cast(tf.io.parse_tensor(example["spk"],out_type=tf.float32),dtype=tf.float32)
    return spe,wav,ppg,pit,spk

def load_dataset():
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        "test.tfrecords"
    )  # automatically interleaves reads from multiple files
    # test = dataset.as_numpy_iterator()
    # for i in test:
    #     temp = read_tfrecord(i)
    # dataset = dataset.with_options(
    #     ignore_order
    # )  # uses data as soon as it streams in, rather than in its original order
    # dataset = dataset.map(
    #     read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
    # )
    return dataset
    

def get_dataset():
    dataset = load_dataset()
    # dataset = dataset.shuffle(2)
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    # dataset = dataset.batch(1)
    return dataset
class GANModel(tf.keras.Model):
    def __init__(self, discriminator, generator, stft,hp):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        #self.latent_dim = latent_dim
        self.stft=stft
        self.hp = hp
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
    def train_step(self, data):
            #spec,audio,ppg,pit,spk = data
            spec = data[0][0]
            audio = data[0][1]
            ppg = data[0][2]
            pit = data[0][3]
            spk = data[0][4]
            #audio = tf.squeeze(wav,0)
            # len_pit = pit.shape[1]
            # len_ppg = ppg.shape[1]
            # len_min = min(len_pit,len_ppg)  
            # pit = pit[:len_min]
            # ppg = ppg[:len_min, :]
            # spec = spec[:,:len_min,: ]        
            ppg_l = ppg.shape[1]
            spec_l =spec.shape[1]
            with tf.GradientTape(persistent= True) as tape:
                fake_audio, ids_slice, z_mask, \
                    (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = self.generator(
                        ppg, pit, spec, spk, ppg_l, spec_l,training=True)
                audio = commons.slice_segments(
                    audio, ids_slice * self.hp.data.hop_length, self.hp.data.segment_size)  # slice
                #spk_loss = vpr_loss(spk, spk_preds, tf.cast(spk_preds.size(0),tf.float32).fill_(1.0))
                fake_audio = tf.expand_dims(fake_audio,1)
                mel_fake = self.stft.mel_spectrogram(tf.expand_dims(fake_audio,1))
                mel_real = self.stft.mel_spectrogram(tf.expand_dims(audio,1))

                mel_loss = tf.keras.losses.MAE(mel_fake, mel_real) * self.hp.train.c_mel
                stft_losses = []
                for fs, ss, wl in eval(self.hp.mrd.resolutions):
                    stft_losses += [STFTLoss(fs, ss, wl)]
                #sc_mag_loss = stft_criterion(fake_audio, audio)
                sc_mag_loss =0.0
                for f in stft_losses:
                    sc_mag_loss += f(fake_audio, audio)

                sc_mag_loss /= len(stft_losses)
                stft_loss = sc_mag_loss * self.hp.train.c_stft
                disc_fake = self.discriminator(fake_audio)
                score_loss = 0.0
                for (_, score_fake) in disc_fake:
                    score_loss += tf.reduce_mean(tf.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(disc_fake)


                disc_real = self.discriminator(audio)
                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += tf.reduce_mean(tf.abs(fake - real))
                feat_loss = feat_loss / len(disc_fake)
                feat_loss = feat_loss * 2
        
                # Loss
                loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * self.hp.train.c_kl
                loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * self.hp.train.c_kl

                loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f + loss_kl_r * 0.5 
                disc_fake = self.discriminator(tf.stop_gradient(fake_audio),training=True)
                disc_real = self.discriminator(audio,training=True)

                loss_d = 0.0
                for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                    loss_d += tf.reduce_mean(tf.pow(score_real - 1.0, 2))
                    loss_d += tf.reduce_mean(tf.pow(score_fake, 2))
                loss_d = loss_d / len(disc_fake)
            g_gradients = tape.gradient(loss_g, self.generator.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            self.g_optimizer.apply_gradients(grads_and_vars=list(zip(g_gradients, self.generator.trainable_weights)))
            d_gradients = tape.gradient(loss_d, self.discriminator.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            self.d_optimizer.apply_gradients(grads_and_vars=zip(d_gradients, self.discriminator.trainable_weights))
            self.gen_loss_tracker.update_state(loss_g)
            self.disc_loss_tracker.update_state(loss_d)
            return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            }
            # loss_g = loss_g
            # loss_d = loss_d
            # loss_s = stft_loss
            # loss_m = mel_loss
            # loss_k = loss_kl_f
            # loss_r = loss_kl_r
        
            # print("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f | step %d" % (
            #     loss_g, loss_m, loss_s, loss_d, loss_k, loss_r,step))
def train(rank, args, chkpt_path, hp, hp_str):
    
    #try TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    # policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    strategy = tf.distribute.TPUStrategy(resolver)
    tf.random.set_seed(hp.train.seed)
    #tnp.experimental_enable_numpy_behavior()
    # train_dataset = strategy.distribute_datasets_from_function(
    # lambda _: get_dataset())
    with strategy.scope():
        d_optimizer=tf.keras.optimizers.AdamW(learning_rate=hp.train.learning_rate, beta_1=hp.train.betas[0],beta_2=hp.train.betas[1], epsilon=hp.train.eps)
        g_optimizer=tf.keras.optimizers.AdamW(learning_rate=hp.train.learning_rate, beta_1=hp.train.betas[0],beta_2=hp.train.betas[1], epsilon=hp.train.eps)
        stft = TacotronSTFT(filter_length=hp.data.filter_length,
                            hop_length=hp.data.hop_length,
                            win_length=hp.data.win_length,
                            n_mel_channels=hp.data.mel_channels,
                            sampling_rate=hp.data.sampling_rate,
                            mel_fmin=hp.data.mel_fmin,
                            mel_fmax=hp.data.mel_fmax,
                            center=False)
        model_g = SynthesizerTrn(
        hp.data.filter_length // 2 + 1,
        hp.data.segment_size // hp.data.hop_length,
        hp)
            
        model_d=Discriminator(hp)
        cond_gan = GANModel(discriminator=model_d, generator=model_g,stft=stft,hp=hp)
        cond_gan.compile(
            d_optimizer=d_optimizer,
            g_optimizer=g_optimizer,
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True)
         
        )
    dataset = get_dataset()
    test = dataset.as_numpy_iterator()
    # "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
    #     "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
    #     "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
    #     "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
    #     "spk":
    spe_list = []
    wav_list = []
    ppg_list = []
    pit_list = []
    spk_list = []
    for i in test:
        temp = read_tfrecord(i)
      #  test = tf.zeros_like([400,513]).numpy()
        # for i in range(400):
        #     test[i]=temp[0][i]
        spec = temp[0][:400,:]
        spe_list.append(spec)
        wav_list.append(tf.squeeze(temp[1],0))
        ppg_list.append(temp[2])
        pit_list.append(temp[3])
        spk_list.append(temp[4])
    spe_list = tf.keras.preprocessing.sequence.pad_sequences(
    spe_list, padding="post",dtype="float32"
    )
    spe_list=tf.convert_to_tensor(spe_list)
    wav_list = tf.keras.preprocessing.sequence.pad_sequences(
    wav_list, padding="post",dtype="float32"
    )
    wav_list=tf.convert_to_tensor(wav_list)
    ppg_list = tf.keras.preprocessing.sequence.pad_sequences(
    ppg_list, padding="post",dtype="float32"
    )
    ppg_list=tf.convert_to_tensor(ppg_list)
    pit_list = tf.keras.preprocessing.sequence.pad_sequences(
    pit_list, padding="post",dtype="float32"
    )
    pit_list=tf.convert_to_tensor(pit_list)
    spk_list=tf.convert_to_tensor(spk_list)
    #tf.config.run_functions_eagerly(True)
    cond_gan.fit([spe_list,wav_list,ppg_list,pit_list,spk_list],batch_size=8, epochs=20)
                    
