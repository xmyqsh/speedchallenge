import tensorflow as tf
import loader

training_dataset = loader.load_tfrecord("/mnt/Bulk/commaai/monolithic_train.tfrecord", True)
validation_dataset = loader.load_tfrecord("/mnt/Bulk/speedchallenge/monolithic_test.tfrecord", False)

model = tf.keras.Sequential()

# encoder
model.add(tf.keras.layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())

# thingy
model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu))

# pose extractor
model.add(tf.keras.layers.Conv2D(6, (1, 1), padding='valid'))
model.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, [1, 2])))
model.add(tf.keras.layers.Reshape((6,)))
model.add(tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :3] * 0.01, x[:, 3:6] * 0.001], axis=1)))

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=['mse'])

model.fit(training_dataset, epochs=10, validation_data=validation_dataset)
