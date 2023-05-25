
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from .nsf import SourceModuleHnNSF
from .bigv import AMPBlock
#import tensorflow_probability as tfp


class SpeakerAdapter(tf.keras.layers.Layer):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = tf.keras.layers.Dense(
            units=self.adapter_dim,
            activation=None,
            input_shape=(self.speaker_dim,))
        self.W_bias = tf.keras.layers.Dense(
           input_shape=(self.speaker_dim,), 
            units=self.adapter_dim,
            activation=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.W_scale.kernel_initializer=tf.keras.initializers.Zeros()
        self.W_scale.bias_initializer=tf.keras.initializers.Constant(1.)
        self.W_bias.kernel_initializer=tf.keras.initializers.Zeros()
        self.W_bias.bias_initializer=tf.keras.initializers.Zeros()

    def call(self, x, speaker_embedding,training=False):
        mean = tf.reduce_mean(x,axis=-1, keepdims=True)
        var = tf.reduce_mean(((x - mean) ** 2),axis=-1, keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding,training=training)
        bias = self.W_bias(speaker_embedding,training=training)
        y *= tf.expand_dims(scale,1)
        y += tf.expand_dims(bias,1)
        return y


class Generator(tf.keras.layers.Layer):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.hp = hp
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        self.adapter = SpeakerAdapter(hp.vits.spk_dim, hp.gen.upsample_input)
        # pre conv
        self.conv_pre = tf.keras.layers.Conv1D(hp.gen.upsample_initial_channel, 7, 1, padding='same')
        # nsf
        #self.f0_upsamp = tf.keras.layers.UpSampling1D(size=3294)
           # size=np.prod(hp.gen.upsample_rates))
        #self.m_source = SourceModuleHnNSF(sampling_rate=hp.data.sampling_rate)
        self.noise_convs = []#nn.ModuleList()
        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = []#nn.ModuleList()
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            self.ups.append(
               #tfp.layers.weight_norm.WeightNorm(
                    tf.keras.layers.Conv1DTranspose(
                       # hp.gen.upsample_initial_channel // (2 ** i),
                        filters=hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        strides=u,
                        padding='same')
               # )
            )
            # nsf
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    tf.keras.layers.Conv1D(
                       # 1,
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        strides=stride_f0,
                       # padding=stride_f0 // 2,
                       padding='same'
                    )
                )
            else:
                self.noise_convs.append(
                    tf.keras.layers.Conv1D(
                  #  1, 
                    hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=1)
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = []#nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                self.resblocks.append(AMPBlock(ch, k, d))

        # post conv
        self.conv_post = tf.keras.layers.Conv1D(
            1, 7, 1, 
           padding='same',
            use_bias=False)
        # weight initialization
        # for conv in self.ups:
        #     init_weights(conv)
       # self.ups.apply(init_weights)

    def call(self, spk, x, f0,training=False):
        # adapter
        #x = self.adapter(x, spk,training=training)
        # nsf
        #f0 = f0[:,:, None]    
        #f0 =self.f0_upsamp(f0,training=training)
        #har_source = self.m_source(f0,training=training)
        #har_source = tf.transpose(har_source,[0,2,1])
        
        x = self.conv_pre(x,training=training)
        #x = x.to_tensor()
        for i in range(self.num_upsamples):
            x = tf.keras.layers.LeakyReLU(0.1)(x)
            # upsampling
            x = self.ups[i](x,training=training)
            # nsf
            #x_source = self.noise_convs[i](har_source,training=training)
            #x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,training=training)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,training=training)
            x = xs / self.num_kernels

        # post conv
        x = tf.keras.layers.LeakyReLU(0.01)(x)

        x = self.conv_post(x,training=training)

        x = tf.tanh(x)
        #x=tf.transpose(x,[0,2,1])

        return x

    # def remove_weight_norm(self):
    #     for l in self.ups:
    #         remove_weight_norm(l)
    #     for l in self.resblocks:
    #         l.remove_weight_norm()

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        # if inference:
        #     self.remove_weight_norm()

    def pitch2source(self, f0):
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)  # [1,len,1]
        har_source = self.m_source(f0)
        har_source = har_source.transpose(1, 2)  # [1,1,len]
        return har_source

    def source2wav(self, audio):
        MAX_WAV_VALUE = 32768.0
        audio = audio.squeeze()
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio.cpu().detach().numpy()

    def inference(self, spk, x, har_source):
        # adapter
        x = self.adapter(x, spk)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = tf.keras.layers.LeakyReLU(0.1)(x)
            # upsampling
            x = self.ups[i](x)
            # nsf
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = tf.keras.layers.LeakyReLU(0.01)(x)
        x = self.conv_post(x)
        x = tf.tanh(x)
        return x
