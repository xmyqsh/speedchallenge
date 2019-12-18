import tensorflow as tf

image_feature_description = {
    'frame_one': tf.FixedLenFeature([], tf.string),
    'frame_two': tf.FixedLenFeature([], tf.string),
    'frame_three': tf.FixedLenFeature([], tf.string),
    'frame_four': tf.FixedLenFeature([], tf.string),
    'plus_one_position': tf.FixedLenFeature([3], tf.float32),
    'plus_one_orientation': tf.FixedLenFeature([3], tf.float32),
    'plus_two_position': tf.FixedLenFeature([3], tf.float32),
    'plus_two_orientation': tf.FixedLenFeature([3], tf.float32),
    'plus_three_position': tf.FixedLenFeature([3], tf.float32),
    'plus_three_orientation': tf.FixedLenFeature([3], tf.float32),
    'speed': tf.FixedLenFeature([], tf.float32),
}

def decode_and_process_frame(frame, mirror, training):
    image = tf.image.decode_jpeg(frame, channels=3)
    if training:
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)
    if mirror:
        image = tf.image.flip_left_right(image)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    return image

def parse_record(tfrecord, training):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    if training:
        mirror = tf.random_uniform([]) < 0.5
    else:
        mirror = False

    frame_one = decode_and_process_frame(proto['frame_one'], mirror, training)
    frame_two = decode_and_process_frame(proto['frame_two'], mirror, training)
    position = proto['plus_one_position']
    orienation = proto['plus_one_orientation']
    speed = proto['speed']

    image = tf.concat((frame_one, frame_two), axis=2)

    if mirror:
        position = (1, -1, 1) * position
        orienation = (-1, 1, -1) * orienation

    if not training:
        return tf.data.Dataset.from_tensors((image, position, orienation, speed))

    rev_image = tf.concat((frame_two, frame_one), axis=2)
    rev_position = -1 * position
    rev_orientation = -1 * orienation
    
    images = tf.stack((image, rev_image))
    positions = tf.stack((position, rev_position))
    orientations = tf.stack((orienation, rev_orientation))
    speeds = tf.stack((speed, speed))

    return tf.data.Dataset.from_tensor_slices((images, positions, orientations, speeds))

def load_tfrecord(filename, training):
    dataset = tf.data.TFRecordDataset(filename)

    if training:
        dataset = dataset.shuffle(300000)
    dataset = dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.flat_map(lambda x: x)
    dataset = dataset.batch(200)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
