import copy
import math
import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
import tensorflow as tf
from vits import commons
#from vits.modules import LayerNorm


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        **kwargs
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = tf.keras.layers.Dropout(p_dropout,dtype=tf.float32)
        self.attn_layers = []#nn.ModuleList()
        self.norm_layers_1 = []#nn.ModuleList()
        self.ffn_layers = []#nn.ModuleList()
        self.norm_layers_2 = []#nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                tf.keras.layers.MultiHeadAttention(
                    output_shape=hidden_channels,
                    key_dim=2,
                    num_heads=n_heads,
                    dropout=p_dropout,
                    dtype=tf.float32
                )
            )
            self.norm_layers_1.append(tf.keras.layers.LayerNormalization(dtype=tf.float32))
            self.ffn_layers.append(
            FFN(
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(tf.keras.layers.LayerNormalization(dtype=tf.float32))

    def call(self, x, x_mask,training=False):
        x = tf.cast(x,tf.float32)
        x_mask = tf.cast(x_mask,tf.float32)
        attn_mask = tf.expand_dims(x_mask,-1) * tf.expand_dims(x_mask,0)
        attn_mask = tf.expand_dims(attn_mask,1)
        attn_mask = tf.squeeze(attn_mask,-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](query=x, value=x,key=x,attention_mask=attn_mask,training=training)
            y = self.drop(y,training=training)
            x = self.norm_layers_1[i](x + y,training=training)
            #x = x + y
            y = self.ffn_layers[i](x, x_mask,training=training)
          
            y = self.drop(y,training=training)
            x = self.norm_layers_2[i](x + y,training=training)  
            #x = x + y     
        x = x * x_mask
        
        return tf.cast(x,tf.bfloat16)

class FFN(tf.keras.layers.Layer):
    def __init__(
        self,
       # in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
      #  self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = 'causal'
        else:
            self.padding = 'same'

        self.conv_1 = tf.keras.layers.Conv1D(filter_channels, kernel_size,padding=self.padding)
        self.conv_2 = tf.keras.layers.Conv1D(out_channels, kernel_size,padding=self.padding)
        self.drop = tf.keras.layers.Dropout(p_dropout)

    def call(self, x, x_mask,training=False):
        x_mask = tf.cast(x_mask,tf.bfloat16)
        x = x * x_mask
        x = self.conv_1(x,training=training)
       
        if self.activation == "gelu":
            x = x * tf.sigmoid(1.702 * x)
        else:
            x = tf.keras.layers.ReLU()(x)
        x = self.drop(x,training=training)
        x = x * x_mask
        x = self.conv_2(x,training=training)
        x = x * x_mask
        return x
