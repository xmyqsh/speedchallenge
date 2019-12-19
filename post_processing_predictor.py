import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy as np
import post_processing_loader
import os
import nets

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")

training_dataset = post_processing_loader.load_tfrecord("D:\\commaai\\monolithic_train.tfrecord", True, False, False)
validation_dataset = post_processing_loader.load_tfrecord("D:\\speedchallenge\\monolithic_final.tfrecord", False, False, False)
validation_dataset_backward = post_processing_loader.load_tfrecord("D:\\speedchallenge\\monolithic_final.tfrecord", False, True, False)
validation_dataset_mirrored = post_processing_loader.load_tfrecord("D:\\speedchallenge\\monolithic_final.tfrecord", False, False, True)
validation_dataset_backward_mirrored = post_processing_loader.load_tfrecord("D:\\speedchallenge\\monolithic_final.tfrecord", False, True, True)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)
validation_backward_init_op = iterator.make_initializer(validation_dataset_backward)
validation_mirrored_init_op = iterator.make_initializer(validation_dataset_mirrored)
validation_backward_mirrored_init_op = iterator.make_initializer(validation_dataset_backward_mirrored)

frames, positions, orienations, speeds = iterator.get_next()

gt_pose = tf.concat((positions, orienations), axis=1)

training = tf.placeholder(tf.bool)

# conv1 = tf.layers.conv2d(frames, 32, (7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu)
# conv1 = tf.layers.batch_normalization(conv1, training=training)
# conv2 = tf.layers.conv2d(conv1, 64, (5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu)
# conv2 = tf.layers.batch_normalization(conv2, training=training)
# conv3 = tf.layers.conv2d(conv2, 128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
# conv3 = tf.layers.batch_normalization(conv2, training=training)
# conv4 = tf.layers.conv2d(conv3, 256, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
# conv4 = tf.layers.batch_normalization(conv4, training=training)
# conv5 = tf.layers.conv2d(conv4, 512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)
# conv5 = tf.layers.batch_normalization(conv5, training=training)
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

# validation loss is the predicted at the middle frame
predicted_speed = 20 * tf.norm(pose[:, :3], axis=1)

speed_loss = tf.losses.mean_squared_error(speeds, predicted_speed)

corr = tfp.stats.correlation(speeds, predicted_speed, sample_axis=0, event_axis=None)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.checkpoint_dir:
        checkpoint_dir = FLAGS.checkpoint_dir
        print('attempting to load checkpoint from {}'.format(checkpoint_dir))
        
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        checkpoint_dir = 'checkpoints/{}'.format(time.strftime("%m_%d_%y-%H_%M"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('no checkpoint specified, starting training from scratch')


    # labels = np.loadtxt("data/train.txt")
    # print(len(labels))

    sess.run(validation_init_op)
    forward_predictions = []
    while True:
        try:
            ss, pss = sess.run((speeds, predicted_speed), {training: False})
            forward_predictions.extend(pss)
        except tf.errors.OutOfRangeError:
            break

    sess.run(validation_backward_init_op)
    backward_predictions = []
    while True:
        try:
            ss, pss = sess.run((speeds, predicted_speed), {training: False})
            backward_predictions.extend(pss)
        except tf.errors.OutOfRangeError:
            break

    sess.run(validation_mirrored_init_op)
    mirrored_predictions = []
    while True:
        try:
            ss, pss = sess.run((speeds, predicted_speed), {training: False})
            mirrored_predictions.extend(pss)
        except tf.errors.OutOfRangeError:
            break

    sess.run(validation_backward_mirrored_init_op)
    backward_mirrored_predictions = []
    while True:
        try:
            ss, pss = sess.run((speeds, predicted_speed), {training: False})
            backward_mirrored_predictions.extend(pss)
        except tf.errors.OutOfRangeError:
            break

    predictions = np.mean([forward_predictions, backward_predictions, mirrored_predictions, backward_mirrored_predictions], axis=0)
    predictions = np.convolve(predictions, np.ones((5,))/5, mode='same')
    predictions = np.insert(predictions, 0, predictions[0])
    print(len(predictions))
    # # print(np.square(np.subtract(predictions, labels)).mean())
    # with open("test.txt", "w") as f:
    #     f.write()
    np.savetxt("test.txt", predictions, fmt='%.6f')
