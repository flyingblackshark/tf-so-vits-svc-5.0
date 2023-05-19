# Adapted from https://github.com/ubisoft/ubisoft-laforge-daft-exprt Apache License Version 2.0
# Unsupervised Domain Adaptation by Backpropagation

import tensorflow as tf


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


class SpeakerClassifier(tf.keras.layers.Layer):

    def __init__(self, embed_dim, spk_dim):
        super(SpeakerClassifier, self).__init__()
        self.classifier = tf.keras.Sequential(
            GradReverse(),
            tf.keras.layers.BatchNormalization(tf.keras.layers.Conv1D(embed_dim, kernel_size=5, padding=2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(tf.keras.layers.Conv1D(embed_dim, kernel_size=5, padding=2)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(tf.keras.layers.Conv1D(spk_dim, kernel_size=5, padding=2))
        )

    def call(self, x):
        ''' Forward function of Speaker Classifier:
            x = (B, embed_dim, len)
        '''
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        outputs = tf.reduce_mean(outputs, axis=-1)
        return outputs
