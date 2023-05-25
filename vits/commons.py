import tensorflow as tf
import numpy as np
# def clip_grad_value_(parameters, clip_value, norm_type=2):
#     if isinstance(parameters, tf.Tensor):
#         parameters = [parameters]
#     parameters = list(filter(lambda p: p.grad is not None, parameters))
#     norm_type = float(norm_type)
#     if clip_value is not None:
#         clip_value = float(clip_value)

#     total_norm = 0
#     for p in parameters:
#         param_norm = p.grad.data.norm(norm_type)
#         total_norm += param_norm.item() ** norm_type
#         if clip_value is not None:
#             p.grad.data.clamp_(min=-clip_value, max=clip_value)
#     total_norm = total_norm ** (1.0 / norm_type)
#     return total_norm
# def mo(input_tensor, position=None, value=None):
#     input_tensor = input_tensor.numpy()
#     input_tensor[tuple(position)] = value
#     return input_tensor
@tf.function
def slice_segments(x, ids_str, segment_size=4):
    ret = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(tf.shape(x)[0]):
        idx_str = ids_str[i]
        ret.write(i,tf.slice(x,[i,idx_str,0],[1,segment_size,x.shape[2]]))
    return ret.stack()
@tf.function
def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    for i in range(tf.shape(x)[0]):
        idx_str = ids_str[i]
        ret.write(i,tf.slice(x,[i,idx_str],[1,segment_size]))
    return ret.stack()
@tf.function
def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
    b, d, t = tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[2]

    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = tf.cast((tf.random.stateless_uniform(shape=[b],seed=[1,2]) * ids_str_max),dtype=tf.int32)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str
# def sequence_mask(length, max_length=None):
#     if max_length is None:
#         max_length = length.max()
#     x = tf.range(0,max_length)
#     return tf.math.less(tf.expand_dims(x,0) , tf.expand_dims(length,1))