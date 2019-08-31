import tensorflow as tf
import time
import numpy as np

image_feature_description = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.float32),
}

def parse_record(tfrecord):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    image_decoded = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image_flipped = tf.image.random_flip_left_right(image_decoded)
    image_hue = tf.image.random_hue(image_flipped, 0.08)
    image_sat = tf.image.random_saturation(image_hue, 0.6, 1.6)
    image_bright = tf.image.random_brightness(image_sat, 0.05)
    image_contrast = tf.image.random_contrast(image_bright, 0.7, 1.3)

    image_float = tf.image.convert_image_dtype(image_contrast, tf.float32)

    return image_float, proto['label']

def load_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(parse_record)
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(5))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(20)
    return dataset

training_dataset = load_dataset("D:\\speedchallenge\\temporal\\train.tfrecords")
validation_dataset = load_dataset("D:\\speedchallenge\\temporal\\validation.tfrecords")

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# frame is 5 x 640 x 480 x 3 pixels, speed is a float
frames, speeds = iterator.get_next()

# 5x640x480x3 -> 5x320x240x8
conv1 = tf.layers.conv3d(frames, 8, (3, 3, 3), strides = (1, 2, 2), padding='same')
# 5x320x240x8 -> 5x160x120x16
conv2 = tf.layers.conv3d(conv1, 16, (3, 3, 3), strides = (1, 2, 2), padding='same')
# 5x160x120x16 -> 5x80x60x32
conv3 = tf.layers.conv3d(conv2, 32, (3, 3, 3), strides = (1, 2, 2), padding='same')
# 5x80x60x32 -> 5x40x30x64
conv4 = tf.layers.conv3d(conv3, 64, (3, 3, 3), strides = (1, 2, 2), padding='same')
# 5x40x30x64 -> 5x20x15x64
conv5 = tf.layers.conv3d(conv4, 64, (3, 3, 3), strides = (1, 2, 2), padding='same')

out = tf.reshape(conv5, (-1, 5*20*15*64))
out = tf.layers.dense(out, 1)

# predict the speed at the middle frame
speed = speeds[:,2]
speed = tf.expand_dims(speed, 1)
loss = tf.losses.mean_squared_error(speed, out)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

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
                l, _ = sess.run((loss, train_step))
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
                validation_loss = sess.run(loss)
                validation_losses.append(validation_loss)
            except tf.errors.OutOfRangeError:
                break
        print('\n\nmse after {} epochs: {:.4f}\n\n'.format(epoch, np.mean(validation_losses)))
