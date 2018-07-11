import tensorflow as tf


def read_and_decode(filename):
    # generate a queue with a given file name
    print("reading tfrecords from {}".format(filename))
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return the file and the name of file
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                           features={
                                               'label': tf.VarLenFeature(tf.int64),
                                               'img_raw': tf.FixedLenFeature([],tf.string),

                                           })
    # You can do more image distortion here for training data
    label = tf.cast(features['label'], tf.int64)
    img = tf.decode_raw(features['img_raw'], tf.uint8)  # notice the type of data
    img = tf.reshape(img, [32,280, 1])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    return img, label