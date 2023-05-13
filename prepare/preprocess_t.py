import tensorflow as tf
raw_dataset = tf.data.TFRecordDataset("./test.tfrecords")
feature={
            "spe": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "wav": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "ppg": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "pit": tf.io.FixedLenFeature([], tf.string, default_value=''),
            "spk": tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
def _parse_function(example_proto):
# Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature)
parsed_dataset = raw_dataset.map(_parse_function)
for s in parsed_dataset:
    test = tf.io.parse_tensor(s['spe'],out_type=tf.float32)
    print(test)


#raw_dataset = tf.data.Dataset.from_tensor_slices("./test.tfrecords")

# new_dataset_list = []
# for s in parsed_dataset:
#     res = [tf.cast(tf.io.parse_tensor(s["spe"],out_type=tf.float32),dtype=tf.float64),
#         tf.cast(tf.io.parse_tensor(s["wav"],out_type=tf.float32),dtype=tf.float64),
#         tf.cast(tf.io.parse_tensor(s["ppg"],out_type=tf.float32),dtype=tf.float64),
#         tf.io.parse_tensor(s["pit"],out_type=tf.float64),
#         tf.cast(tf.io.parse_tensor(s["spk"],out_type=tf.float32),dtype=tf.float64)]
#     new_dataset_list.append(res)
# new_dataset = tf.data.Dataset.from_tensor_slices(new_dataset_list)
# tf.data.Dataset.save(new_dataset,"new_dataset")
# new_dataset = new_dataset.batch(2)
# for element in new_dataset:
#     print(tf.io.parse_tensor(element[0][4],out_type=tf.float32))