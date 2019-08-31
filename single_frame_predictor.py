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
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)

    return image_float, proto['label']

def load_dataset(filename):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(parse_record)
    dataset = dataset.batch(10)
    return dataset

training_dataset = load_dataset("D:\\speedchallenge\\temporal\\train.tfrecords")
validation_dataset = load_dataset("D:\\speedchallenge\\temporal\\validation.tfrecords")

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# frame is 640 x 480 pixels, speed is a float
frame, speed = iterator.get_next()

def residual_block(input, channels, downsample):
    shortcut = input
    strides = (1, 1)
    if downsample:
        strides = (2, 2)
        shortcut = tf.layers.conv2d(input, channels, (1, 1), strides=strides, padding='same')
        shortcut = tf.layers.batch_normalization(shortcut, training=True)
        
    conv1 = tf.layers.conv2d(input, channels, (3, 3), strides=(1, 1), padding='same')
    conv1 = tf.layers.batch_normalization(conv1, training=True)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d(conv1, channels, (3, 3), strides=strides, padding='same')
    conv2 = tf.layers.batch_normalization(conv2, training=True)


    conv2 += shortcut
    output = tf.nn.relu(conv2)

    return output


out = frame

# 640x480x3 -> 20x15x64
for i in range(5):
    out = residual_block(out, 64, True)

out = tf.reshape(out, (-1, 20*15*64))
out = tf.layers.dense(out, 1)

speed = tf.expand_dims(speed, 1)
loss = tf.losses.mean_squared_error(speed, out)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 31):
        sess.run(training_init_op)
        i = 0
        previous_step_time = time.time()
        while True:
            try:
                i += 1
                l, _ = sess.run((loss, train_step))
                if i % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    print('epoch {:d} - step {:d} - time {:.2f}s : loss {:.4f}'.format(epoch, i, time_elapsed, l))
                    previous_step_time = current_step_time
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
