import tensorflow as tf

image_feature_description = {
    'frame_one': tf.io.FixedLenFeature([], tf.string),
    'frame_two': tf.io.FixedLenFeature([], tf.string),
    'frame_three': tf.io.FixedLenFeature([], tf.string),
    'frame_four': tf.io.FixedLenFeature([], tf.string),
    'plus_one_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_one_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'plus_two_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_two_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'plus_three_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_three_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'speed': tf.io.FixedLenFeature([], tf.float32),
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
    proto = tf.io.parse_single_example(tfrecord, image_feature_description)

    if training:
        mirror = tf.random.uniform([]) < 0.5
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

    pose = tf.concat((position, orienation), axis=0)

    if not training or tf.random.uniform([]) < 0.5:
        return {'frames': image}, {'pose': pose, 'speed': [speed]}
    else:
        rev_image = tf.concat((frame_two, frame_one), axis=2)
        rev_pose = -1 * pose
        return {'frames': rev_image}, {'pose': rev_pose, 'speed': [speed]}

def load_tfrecord(filename, batch_size, training):
    dataset = tf.data.TFRecordDataset(filename)

    if training:
        dataset = dataset.shuffle(300000)
    dataset = dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
