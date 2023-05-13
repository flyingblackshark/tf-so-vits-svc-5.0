
import tensorflow as tf
import tensorflow_addons as tfa
from vits.losses import fused_add_tanh_sigmoid_multiply
class Flip(tf.keras.Model):
    def call(self, x, *args, reverse=False, **kwargs):   
        x = tf.reverse(x, [1])
        logdet = tf.zeros([x.shape[0]])#.to(dtype=x.dtype, device=x.device)
        return x, logdet
class WN(tf.keras.Model):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = []#torch.nn.ModuleList()
        self.res_skip_layers = []#torch.nn.ModuleList()
        self.drop = tf.keras.layers.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = tf.keras.layers.Conv1D(
               # gin_channels,
                2 * hidden_channels * n_layers, kernel_size=1
            )
            #self.cond_layer = tfa.layers.WeightNormalization(cond_layer)
            
        for i in range(n_layers):
            dilation = dilation_rate**i
            #padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = tf.keras.layers.Conv1D(
              #  hidden_channels,
                filters=2 * hidden_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='same'
                #padding=padding,
            )
            in_layer = tfa.layers.WeightNormalization(in_layer)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = tf.keras.layers.Conv1D(res_skip_channels, 1)
            res_skip_layer = tfa.layers.WeightNormalization(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def call(self, x, x_mask, g=None, **kwargs):
        output = tf.zeros_like(x)
        n_channels_tensor = tf.cast([self.hidden_channels],dtype=tf.int32)

        if g is not None:
            g = tf.transpose(g,[0,2,1])
            g = self.cond_layer(g)
            g = tf.transpose(g,[0,2,1])

        for i in range(self.n_layers):
            temp = tf.transpose(x,[0,2,1])
            x_in = self.in_layers[i](temp)
            temp = tf.transpose(x_in,[0,2,1])
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = tf.zeros_like(temp)

            acts = fused_add_tanh_sigmoid_multiply(temp, g_l, n_channels_tensor)
            acts = self.drop(acts)
            acts = tf.transpose(acts,[0,2,1])
            res_skip_acts = self.res_skip_layers[i](acts)
            res_skip_acts = tf.transpose(res_skip_acts,[0,2,1])
            
            if i < self.n_layers - 1:
               
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                temp = x + res_acts
                #temp = tf.transpose(temp,[0,2,1])
                x =  temp * x_mask
                #x = tf.transpose(x,[0,2,1])
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                #res_skip_acts=tf.transpose(res_skip_acts,[0,2,1])
                output = output + res_skip_acts
                #output=tf.transpose(output,[0,2,1])
        return output * x_mask

class ResidualCouplingLayer(tf.keras.Model):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = tf.keras.layers.Conv1D(hidden_channels, 1)
        # no use gin_channels
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
        )
        self.post = tf.keras.layers.Conv1D(
            #hidden_channels, 
            self.half_channels * (2 - mean_only),
            kernel_size=1)
        self.post.kernel_initializer=tf.keras.initializers.Zeros()
        self.post.bias_initializer=tf.keras.initializers.Zeros()
        # SNAC Speaker-normalized Affine Coupling Layer
        self.snac = tf.keras.layers.Conv1D(2 * self.half_channels, 1)

    def call(self, x, x_mask, g=None, reverse=False):
        temp = tf.expand_dims(g,-1)
        temp = tf.transpose(temp,[0,2,1])
        speaker = self.snac(temp)
        speaker = tf.transpose(speaker,[0,2,1])
        speaker_m, speaker_v = tf.split(speaker,2, axis=1)  # (B, half_channels, 1)
        x0, x1 = tf.split(x, [self.half_channels] * 2, 1)
        # x0 norm
        x0_norm = (x0 - speaker_m) * tf.exp(-speaker_v) * x_mask
        x0_norm = tf.transpose(x0_norm,[0,2,1])
        temp = self.pre(x0_norm)
        temp = tf.transpose(temp,[0,2,1])
        h =  temp * x_mask
        #h = tf.transpose(h,[0,2,1])
        # don't use global condition
        h = self.enc(h, x_mask)
        h = tf.transpose(h,[0,2,1])
        temp = self.post(h)
        temp = tf.transpose(temp,[0,2,1])
        stats =  temp * x_mask
        if not self.mean_only:
            m, logs = tf.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = tf.zeros_like(m)

        if not reverse:
            # x1 norm before affine xform
            x1_norm = (x1 - speaker_m) * tf.exp(-speaker_v) * x_mask
            x1 = (m + x1_norm * tf.exp(logs)) * x_mask
            x = tf.concat([x0, x1], 1)
            # speaker var to logdet
            logdet = tf.math.reduce_sum(logs * x_mask, [1, 2]) - tf.math.reduce_sum(
               tf.broadcast_to(speaker_v,[speaker_v.shape[0],speaker_v.shape[1],logs.shape[-1]]) #speaker_v.expand(-1, -1, logs.shape[-1]) 
                * x_mask, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * tf.exp(-logs) * x_mask
            # x1 denorm before output
            x1 = (speaker_m + x1 * tf.exp(speaker_v)) * x_mask
            x = tf.concat([x0, x1], 1)
            # speaker var to logdet
            logdet = tf.math.reduce_sum(logs * x_mask, [1, 2]) + tf.math.reduce_sum(
              tf.broadcast_to(speaker_v,[speaker_v.shape[0],speaker_v.shape[1],logs.shape[-1]])  #speaker_v.expand(-1, -1, logs.shape[-1]) 
                * x_mask, [1, 2])
            return x, logdet

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
