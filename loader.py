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

    image = tf.concat((frame_one, frame_two), axis=2)
    position = proto['position']
    orienation = proto['orientation']

    # if not training:
    #     tf.data.Dataset.from_tensors([image, position, orienation, proto['speed']])

    rev_image = tf.concat((frame_two, frame_one), axis=2)
    rev_position = -1 * proto['position']
    rev_orientation = -1 * proto['orientation']
    
    images = tf.stack((image, rev_image))
    positions = tf.stack((position, rev_position))
    orientations = tf.stack((orienation, rev_orientation))
    speeds = tf.stack((proto['speed'], proto['speed']))

    return tf.data.Dataset.from_tensor_slices((images, positions, orientations, speeds))

def load_tfrecord(filename, training):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(lambda x: x)
    # if training:
    #     dataset = dataset.shuffle(15000)
    dataset = dataset.batch(200)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
