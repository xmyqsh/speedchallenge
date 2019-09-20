import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

image_feature_description = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.float32),
}

def parse_record(tfrecord):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    image = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image = tf.image.resize(image, (480, 640))
    image = tf.image.crop_to_bounding_box(image, 200, 0, 160, 640)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, proto['label']


for segment in tf.data.Dataset.list_files("/mnt/d/waymo/segments/segment_0.tfrecord"):
    for data in tf.data.TFRecordDataset(segment).take(10):
        image, speed = parse_record(data)
        print(speed)
        print(image.shape)