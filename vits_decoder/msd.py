import tensorflow as tf

class ScaleDiscriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaleDiscriminator, self).__init__()
        self.convs = [
            tf.keras.layers.Conv1D(1, 16, 15, 1, padding=7),
            tf.keras.layers.Conv1D(16, 64, 41, 4, groups=4, padding=20),
            tf.keras.layers.Conv1D(64, 256, 41, 4, groups=16, padding=20),
            tf.keras.layers.Conv1D(256, 1024, 41, 4, groups=64, padding=20),
            tf.keras.layers.Conv1D(1024, 1024, 41, 4, groups=256, padding=20),
            tf.keras.layers.Conv1D(1024, 1024, 5, 1, padding=2)
        ]
        self.conv_post = tf.keras.layers.Conv1D(1024, 1, 3, 1, padding=1)

    def call(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = tf.keras.layers.LeakyReLU(0.1)(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = tf.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]
