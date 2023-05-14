import tensorflow as tf
from vits import commons
from vits.losses import kl_loss
from vits.models import SynthesizerTrn
from vits_decoder.discriminator import Discriminator
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from .tf_gan_model import ConditionalGAN





class Score_Loss(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        score_loss = 0.0
        for (_, score_fake) in y_pred:
            score_loss += tf.math.reduce_mean(tf.pow(score_fake - y_true, 2))
            score_loss = score_loss / len(y_pred)
        return score_loss
class Feat_Loss(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        feat_loss = 0.0
        for (feat_fake, _), (feat_real, _) in zip(y_pred, y_true):
            for fake, real in zip(feat_fake, feat_real):
                feat_loss += tf.math.reduce_mean(tf.abs(fake - real))
                feat_loss = feat_loss / len(y_pred)
                feat_loss = feat_loss * 2
        return feat_loss
class L1_Loss(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        return tf.abs(tf.math.reduce_sum(y_true-y_pred))
class D_Loss(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(y_pred, y_true):
            loss_d += tf.reduce_mean(tf.pow(score_real - 1.0, 2))
            loss_d += tf.reduce_mean(tf.pow(score_fake, 2))
            loss_d = loss_d / len(y_pred)
        return loss_d
def read_tfrecord(example):
    feature=({
        "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "spk": tf.io.FixedLenFeature([], tf.string, default_value=''),
    })
    example = tf.io.parse_single_example(example, feature)
    spe = tf.io.parse_tensor(example["spe"],out_type=tf.float32)
    wav = tf.io.parse_tensor(example["wav"],out_type=tf.float32)
    ppg = tf.io.parse_tensor(example["ppg"],out_type=tf.float32)
    pit = tf.cast(tf.io.parse_tensor(example["pit"],out_type=tf.float64),tf.float32)
    spk = tf.io.parse_tensor(example["spk"],out_type=tf.float32)
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
    
    
   

def train(rank, args, chkpt_path, hp, hp_str):
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
    #self.num_epochs = 201
    l1_loss_fn = L1_Loss()
    d_loss_fn = D_Loss()
    feat_loss_fn =Feat_Loss()
    score_loss_fn = Score_Loss()
    #model = GANModel(hp)
    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (spec,wav,ppg,pit,spk) in enumerate(dataset):
            audio = tf.transpose(wav,[0,1,3,2])
            audio = tf.squeeze(audio,0)
            len_pit = pit.shape[1]
            len_ppg = ppg.shape[1]
            len_min = min(len_pit,len_ppg)  
            pit = pit[:len_min]
            ppg = ppg[:len_min, :]
            spec = spec[:, :len_min]
            ppg_l = ppg.shape[1]
            spec_l =spec.shape[1]
            with tf.GradientTape() as tape:
                fake_audio, ids_slice, z_mask, \
                    (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = model_g(
                        ppg, pit, spec, spk, ppg_l, spec_l,training=True)
                audio = commons.slice_segments(
                    audio, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
                mel_fake = stft.mel_spectrogram(tf.expand_dims(fake_audio,1))
                mel_real = stft.mel_spectrogram(tf.expand_dims(audio,1))
                mel_loss = l1_loss_fn(mel_fake, mel_real) * hp.train.c_mel
                temp1 = tf.expand_dims(fake_audio,1)
                temp2 =tf.expand_dims(audio,1)
            # sc_loss, mag_loss = stft_criterion(temp1,temp2 )
                sc_mag_loss = stft_criterion(temp1,temp2 )
                #stft_loss = (sc_loss + mag_loss) * hp.train.c_stft
                stft_loss = sc_mag_loss * hp.train.c_stft
                res_fake, period_fake, dis_fake = model_d(fake_audio)
                score_loss = score_loss_fn(res_fake + period_fake + dis_fake,1.0)
                res_real, period_real, dis_real = model_d(audio)
                feat_loss = feat_loss_fn(res_fake + period_fake + dis_fake, res_real + period_real + dis_real)
                loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
                loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl
                loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f
            gradients = tape.gradient(loss_g, model_g.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            g_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
                # Loss
        
                
                
            with tf.GradientTape() as tape:
                #optim_d.zero_grad()
                #fake_audio = tf.stop_gradient(fake_audio)
                res_fake, period_fake, dis_fake = model_d(fake_audio)
                res_real, period_real, dis_real = model_d(audio)

                loss_d = d_loss_fn(res_fake + period_fake + dis_fake, res_real + period_real + dis_real)
                tape.gradient(loss_d, model_d.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d_optimizer.apply_gradients(zip(gradients, model_d.trainable_weights))
                #loss_d.backward()
                #clip_grad_value_(model_d.parameters(),  None)
            loss_g = loss_g
            loss_d = loss_d
            loss_s = stft_loss
            loss_m = mel_loss
            loss_k = loss_kl_f
            loss_r = loss_kl_r
            
            print("g %.04f m %.04f s %.04f d %.04f k %.04f r %.04f | step %d" % (
                loss_g, loss_m, loss_s, loss_d, loss_k, loss_r,epoch))
            