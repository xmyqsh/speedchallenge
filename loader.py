import tensorflow as tf

image_feature_description = {
    'frame_one': tf.FixedLenFeature([], tf.string),
    'frame_two': tf.FixedLenFeature([], tf.string),
    'position': tf.FixedLenFeature([3], tf.float32),
    'orientation': tf.FixedLenFeature([3], tf.float32),
    'speed': tf.FixedLenFeature([], tf.float32),
}

def decode_and_process_frame(frame, training):
    image = tf.image.decode_and_crop_jpeg(frame, (200, 0, 160, 640), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image

def parse_record(tfrecord, training):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    frame_one = decode_and_process_frame(proto['frame_one'], training)
    frame_two = decode_and_process_frame(proto['frame_two'], training)

    if (not training) or tf.random.uniform([]) > 0.5:
        image = tf.concat((frame_one, frame_two), axis=2)
        position = proto['position']
        orienation = proto['orientation']
    else:
        image = tf.concat((frame_two, frame_one), axis=2)
        position = -1 * proto['position']
        orienation = -1 * proto['orientation']

    return image, position, orienation, proto['speed']

def load_tfrecord(filename, training):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(15000)
    dataset = dataset.batch(200)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
