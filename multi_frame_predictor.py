import tensorflow as tf
import time
import numpy as np
import loader


training_dataset = loader.load_tfrecord("D:\\commaai\\monolithic_train.tfrecord", True)
validation_dataset = loader.load_tfrecord("D:\\commaai\\monolithic_validation.tfrecord", False)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

frames, positions, orienations, speeds = iterator.get_next()

gt_pose = tf.concat((positions, orienations), axis=1)

conv1 = tf.layers.conv2d(frames, 16, (7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.conv2d(conv1, 32, (5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.conv2d(conv2, 64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
conv4 = tf.layers.conv2d(conv3, 128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
conv5 = tf.layers.conv2d(conv4, 256, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)

pose = tf.layers.conv2d(conv5, 6, (1, 1), padding='valid')
pose = tf.reduce_mean(pose, 2)
pose = tf.reduce_mean(pose, 1)
pose = 0.01 * tf.reshape(pose, (-1, 6))

loss = tf.losses.mean_squared_error(gt_pose, pose)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# validation loss is the predicted at the middle frame
predicted_speed = 20 * tf.norm(pose[:, :3], axis=1)

speed_loss = tf.losses.mean_squared_error(speeds, predicted_speed)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 31):
        sess.run(training_init_op)
        i = 0
        previous_step_time = time.time()
        losses = []
        speed_losses = []
        while True:
            try:
                i += 1
                l, sl, _ = sess.run((loss, speed_loss, train_step))
                losses.append(l)
                speed_losses.append(sl)
                if i % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    print('epoch {:d} - step {:d} - time {:.2f}s : loss {:.4f} speed loss {:.4f}'.format(epoch, i, time_elapsed, np.mean(losses), np.mean(speed_losses)))
                    previous_step_time = current_step_time
                    losses = []
                    speed_losses = []
            except tf.errors.OutOfRangeError:
                break

        sess.run(validation_init_op)
        losses = []
        speed_losses = []
        i = 0
        while True:
            try:
                i += 1
                l, sl = sess.run((loss, speed_loss))
                losses.append(l)
                speed_losses.append(sl)
            except tf.errors.OutOfRangeError:
                break
        print('\n\nafter {} epochs mse: {:.4f}, speed mse: {:.4f}\n\n'.format(epoch, np.mean(losses), np.mean(speed_losses)))
