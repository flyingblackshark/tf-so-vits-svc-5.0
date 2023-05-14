import tensorflow as tf
from vits import commons
from vits.losses import kl_loss
from vits.models import SynthesizerTrn
from vits_decoder.discriminator import Discriminator
from vits_extend.stft import TacotronSTFT
from vits_extend.stft_loss import MultiResolutionSTFTLoss
from .tf_gan_model import ConditionalGAN
# def _normalize_function(example_proto):
#     s = [tf.io.parse_tensor(example_proto['spe'],out_type=tf.float32),
#     tf.io.parse_tensor(example_proto['wav'],out_type=tf.float32),
#     tf.io.parse_tensor(example_proto['ppg'],out_type=tf.float32),
#     tf.io.parse_tensor(example_proto['pit'],out_type=tf.float64),
#     tf.io.parse_tensor(example_proto['spk'],out_type=tf.float32)]
#     return s

feature={
        "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
        "spk": tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature)

def patch_batch(batch):
    # Right zero-pad all one-hot text sequences to max input length
    # mel: [freq, length]
    # wav: [1, length]
    # ppg: [len, 1024]
    # pit: [len]
    # spk: [256]
   #data = data.get_single_element()
    ids_sorted_decreasing = tf.sort(
        tf.cast([x[0].shape[1] for x in batch],dtype=tf.float32), axis=0, direction="DESCENDING"
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
class L1_Loss(tf.keras.losses.Loss):
    def call(self,y_pred,y_true):
        return tf.abs(tf.math.reduce_sum(y_true-y_pred))

def train(rank, args, chkpt_path, hp, hp_str):
    parsed_dataset = tf.data.TFRecordDataset("test.tfrecords").map(_parse_function)
    train_set = []
    for example_proto in parsed_dataset:
        s = [tf.io.parse_tensor(example_proto['spe'],out_type=tf.float32),
        tf.io.parse_tensor(example_proto['wav'],out_type=tf.float32),
        tf.io.parse_tensor(example_proto['ppg'],out_type=tf.float32),
        tf.cast(tf.io.parse_tensor(example_proto['pit'],out_type=tf.float64),dtype=tf.float32),
        tf.io.parse_tensor(example_proto['spk'],out_type=tf.float32)]
        train_set.append(s)
    #parsed_dataset = parsed_dataset.map(_normalize_function)
    #parsed_dataset = parsed_dataset.map(lambda x: tf.ensure_shape(x, shape))
    #parsed_dataset = parsed_dataset.map(_normalize_function)
    #parsed_dataset = parsed_dataset.map(patch_batch)
    #parsed_dataset = parsed_dataset.batch(2)
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
    num_epochs = 201
    l1_loss_fn = L1_Loss()
    # dataset_slice = []
    # ppg_slice = []
    # pit_slice = []
    # spec_slice = []
    # spk_slice = []
    # ppg_l_slice = []
    # spec_l_slice = []
    # for i in train_set:
    #         spec = i[0]
    #         audio =tf.reshape(i[1],[1,1,-1])
    #         ppg = i[2]
    #         pit = i[3]
    #         len_pit = pit.shape[0]
    #         len_ppg = ppg.shape[0]
    #         len_spec = spec.shape[1]
    #         len_min = min(len_pit,len_ppg)
    #         len_min = min(len_min,len_spec)
    #         pit = pit[:len_min]
    #         ppg = ppg[:len_min, :]
    #         spec = spec[:, :len_min]
    #         pit = tf.expand_dims(pit,axis=0)
    #         ppg = tf.expand_dims(ppg,axis=0)
    #         spec = tf.expand_dims(spec,axis=0)
    #         spk = tf.expand_dims(i[4],axis=0)
    #         ppg_l = ppg.shape[1]
    #         spec_l = spec.shape[2]
    #         audio_l = audio.shape[2]
    #         ppg_slice.append(ppg)
    #         pit_slice.append(pit)
    #         spec_slice.append(spec)
    #         spk_slice.append(spk)
    #         ppg_l_slice.append(ppg_l)
    #         spec_l_slice.append(spec_l)
    #         #dataset_slice.append({'ppg':ppg, 'pit':pit,'spec':spec,'spk':spk, 'ppg_l':ppg_l,'spec_l':spec_l})
    # new_datset = tf.data.Dataset.from_tensor_slices(zip(ppg_slice,pit_slice,spec_slice,spk_slice,ppg_l_slice,spec_l_slice))
    model_d.compile()
    for epoch in range(num_epochs):
        # epoch_loss_avg = tf.keras.metrics.Mean()
        # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        for i in train_set:
            spec = i[0]
            audio =tf.reshape(i[1],[1,1,-1])
            ppg = i[2]
            pit = i[3]
            len_pit = pit.shape[0]
            len_ppg = ppg.shape[0]
            len_spec = spec.shape[1]
            len_min = min(len_pit,len_ppg)
            len_min = min(len_min,len_spec)
            pit = pit[:len_min]
            ppg = ppg[:len_min, :]
            spec = spec[:, :len_min]
            pit = tf.expand_dims(pit,axis=0)
            ppg = tf.expand_dims(ppg,axis=0)
            spec = tf.expand_dims(spec,axis=0)
            spk = tf.expand_dims(i[4],axis=0)
            ppg_l = ppg.shape[1]
            spec_l = spec.shape[2]
            audio_l = audio.shape[2]
            #dataset_slice.append([ppg, pit, spec, spk, ppg_l, spec_l])
            with tf.GradientTape(persistent=True) as tape:
                fake_audio, ids_slice, z_mask, \
                    (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r) = model_g(
                        ppg, pit, spec, spk, ppg_l, spec_l,training=True)
                audio = commons.slice_segments(
                    audio, ids_slice * hp.data.hop_length, hp.data.segment_size)  # slice
                mel_fake = stft.mel_spectrogram(tf.expand_dims(fake_audio,1))
                mel_real = stft.mel_spectrogram(tf.expand_dims(audio,1))
                mel_loss = l1_loss_fn(mel_fake, mel_real) * hp.train.c_mel
                #test_mel_loss = loss_fn(mel_fake, mel_real) * hp.train.c_mel
                temp1 = tf.expand_dims(fake_audio,1)
                temp2 =tf.expand_dims(audio,1)
               # sc_loss, mag_loss = stft_criterion(temp1,temp2 )
                sc_mag_loss = stft_criterion(temp1,temp2 )
                #stft_loss = (sc_loss + mag_loss) * hp.train.c_stft
                stft_loss = sc_mag_loss * hp.train.c_stft
                #tape.gradient(stft_loss, model_g.trainable_variables)
                res_fake, period_fake, dis_fake = model_d(fake_audio)
                score_loss = 0.0
                for (_, score_fake) in res_fake + period_fake + dis_fake:
                    score_loss += tf.math.reduce_mean(tf.pow(score_fake - 1.0, 2))
                score_loss = score_loss / len(res_fake + period_fake + dis_fake)
                res_real, period_real, dis_real = model_d(audio)
                feat_loss = 0.0
                for (feat_fake, _), (feat_real, _) in zip(res_fake + period_fake + dis_fake, res_real + period_real + dis_real):
                    for fake, real in zip(feat_fake, feat_real):
                        feat_loss += tf.math.reduce_mean(tf.abs(fake - real))
                feat_loss = feat_loss / len(res_fake + period_fake + dis_fake)
                feat_loss = feat_loss * 2
          
                loss_kl_f = kl_loss(z_f, logs_q, m_p, logs_p, logdet_f, z_mask) * hp.train.c_kl
                loss_kl_r = kl_loss(z_r, logs_p, m_q, logs_q, logdet_r, z_mask) * hp.train.c_kl
                loss_g = score_loss + feat_loss + mel_loss + stft_loss + loss_kl_f
                loss_g_test = score_loss + feat_loss + mel_loss  + loss_kl_f
            # gradients =tape.gradient(score_loss, model_g.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # g_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
            # gradients =tape.gradient(feat_loss, model_g.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # g_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
            # gradients = tape.gradient(mel_loss, model_g.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # g_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
            gradients = tape.gradient(mel_loss, model_g.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            g_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
                # Loss
           
                
                
            with tf.GradientTape() as tape:
                #optim_d.zero_grad()
                fake_audio = tf.stop_gradient(fake_audio)
                res_fake, period_fake, dis_fake = model_d(fake_audio)
                res_real, period_real, dis_real = model_d(audio)

                loss_d = 0.0
                for (_, score_fake), (_, score_real) in zip(res_fake + period_fake + dis_fake, res_real + period_real + dis_real):
                    loss_d += tf.reduce_mean(tf.pow(score_real - 1.0, 2))
                    loss_d += tf.reduce_mean(tf.pow(score_fake, 2))
                loss_d = loss_d / len(res_fake + period_fake + dis_fake)
                tape.gradient(loss_d, model_d.trainable_variables,unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d_optimizer.apply_gradients(zip(gradients, model_g.trainable_weights))
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
            