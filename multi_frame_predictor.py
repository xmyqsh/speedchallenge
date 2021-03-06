import tensorflow as tf
import time
import numpy as np
import loader
import os
import nets

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")

training_dataset = loader.load_tfrecord("/mnt/Bulk/commaai/monolithic_train.tfrecord", True)
validation_dataset = loader.load_tfrecord("/mnt/Bulk/speedchallenge/monolithic_test.tfrecord", False)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

frames, gt_pose, speeds = iterator.get_next()

training = tf.placeholder(tf.bool)

conv5, (conv4, conv3, conv2, conv1) = nets.encoder_resnet(frames, None, training)
conv6 = tf.layers.conv2d(conv5, 512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
conv7 = tf.layers.conv2d(conv6, 512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)

pose = tf.layers.conv2d(conv7, 6, (1, 1), padding='valid')
pose = tf.reduce_mean(pose, [1, 2])
pose = tf.reshape(pose, (-1, 6))
pose = tf.concat([pose[:, :3] * 0.01, pose[:, 3:6] * 0.001], axis=1)

loss = tf.losses.mean_squared_error(gt_pose, pose)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

predicted_speed = 20 * tf.norm(pose[:, :3], axis=1)

speed_loss = tf.losses.mean_squared_error(speeds, predicted_speed)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.checkpoint_dir:
        checkpoint_dir = FLAGS.checkpoint_dir
        print('attempting to load checkpoint from {}'.format(checkpoint_dir))
        
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            print("loading checkpoint {}".format(checkpoint.model_checkpoint_path))
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("couldn't load checkpoint")
            exit()
    else:
        checkpoint_dir = 'checkpoints/{}'.format(time.strftime("%m_%d_%y-%H_%M"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('no checkpoint specified, starting training from scratch')


    for epoch in range(1, 31):
        sess.run(training_init_op)
        i = 0
        previous_step_time = time.time()
        losses = []
        speed_losses = []
        while True:
            try:
                i += 1
                l, sl, _, _ = sess.run((loss, speed_loss, train_step, update_ops), {training: True})
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
                saver.save(sess, checkpoint_dir + "/model", global_step=epoch)
                break

        sess.run(validation_init_op)
        speed_losses = []
        i = 0
        while True:
            try:
                i += 1
                sl = sess.run(speed_loss, {training: False})
                speed_losses.append(sl)
            except tf.errors.OutOfRangeError:
                break
        print('\n\nafter {} epochs speed mse: {:.4f}\n\n'.format(epoch, np.mean(speed_losses)))
