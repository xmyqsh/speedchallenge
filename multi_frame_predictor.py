import tensorflow as tf
import time
import numpy as np

image_feature_description = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.float32),
}

def parse_record(tfrecord, training):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    image = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image = tf.image.crop_to_bounding_box(image, 200, 0, 160, 640)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, proto['label']

def load_dataset(filename, training):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(5))
    if training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(20)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

training_dataset = load_dataset("D:\\speedchallenge\\temporal\\train.tfrecords", True)
validation_dataset = load_dataset("D:\\speedchallenge\\temporal\\validation.tfrecords", False)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# frame is 5 x 640 x 480 x 3 pixels, speed is a float
frames, speeds = iterator.get_next()

training = tf.placeholder(tf.bool)

# 5x640x160x3 -> 5x320x80x8
conv1 = tf.layers.conv3d(frames, 8, (3, 3, 3), strides = (1, 2, 2), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.batch_normalization(conv1, training=training)
# 5x320x80x8 -> 5x160x40x16
conv2 = tf.layers.conv3d(conv1, 16, (3, 3, 3), strides = (1, 2, 2), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.batch_normalization(conv2, training=training)
# 5x160x40x16 -> 5x80x20x32
conv3 = tf.layers.conv3d(conv2, 32, (3, 3, 3), strides = (1, 2, 2), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.batch_normalization(conv3, training=training)
# 5x80x20x32 -> 5x40x10x64
conv4 = tf.layers.conv3d(conv3, 64, (3, 3, 3), strides = (1, 2, 2), padding='same', activation=tf.nn.relu)
conv4 = tf.layers.batch_normalization(conv4, training=training)
# 5x40x10x64 -> 5x20x5x64
conv5 = tf.layers.conv3d(conv4, 64, (3, 3, 3), strides = (1, 2, 2), padding='same', activation=tf.nn.relu)
conv5 = tf.layers.batch_normalization(conv5, training=training)

out = tf.reshape(conv5, (-1, 5*20*5*64))

dropout_rate = tf.placeholder(tf.float32)
out = tf.layers.dropout(out, rate=dropout_rate)

out = tf.layers.dense(out, 1)

# predict the speed at the middle frame
speed = speeds[:,2]
speed = tf.expand_dims(speed, 1)
loss = tf.losses.mean_squared_error(speed, out)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_step = tf.group([train_step, update_ops])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 31):
        sess.run(training_init_op)
        i = 0
        previous_step_time = time.time()
        losses = []
        while True:
            try:
                i += 1
                l, _ = sess.run((loss, train_step), feed_dict={training: True, dropout_rate: 0.4})
                losses.append(l)
                if i % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    print('epoch {:d} - step {:d} - time {:.2f}s : loss {:.4f}'.format(epoch, i, time_elapsed, np.mean(losses)))
                    previous_step_time = current_step_time
                    losses = []
            except tf.errors.OutOfRangeError:
                break

        sess.run(validation_init_op)
        validation_losses = []
        while True:
            try:
                validation_loss = sess.run(loss, feed_dict={training: False, dropout_rate: 0.0})
                validation_losses.append(validation_loss)
            except tf.errors.OutOfRangeError:
                break
        print('\n\nmse after {} epochs: {:.4f}\n\n'.format(epoch, np.mean(validation_losses)))
