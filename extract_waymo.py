import os
import tensorflow as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

for i, segment in enumerate(tf.data.Dataset.list_files('/mnt/d/waymo/*.tfrecord')):
    for j, data in enumerate( tf.data.TFRecordDataset(segment).take(10)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections,
        range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)

        # print(frame.context)

        image = frame.images[0]
        # print(image.name)
        # print(image.velocity)

        speed = np.linalg.norm([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])
        print(speed)
        print(image.pose_timestamp)

        # with open('/mnt/d/waymo/images/segment_{}_frame_{}.jpeg'.format(i, j), 'wb') as f:
        #     f.write(image.image)