import os
import tensorflow as tf
import math
import numpy as np
import itertools
from extract import create_record, write_records
import glob
from multiprocessing import Pool
import cv2

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def extract_segment(segment):
    prev_pose = None
    for i, data in enumerate(tf.data.TFRecordDataset(segment)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        image = frame.images[0]

        pose = np.array(image.pose.transform)
        pose = np.reshape(pose, (4, 4))

        if prev_pose is not None:
            # note: forward, left, up, not forward, right, down!!!
            rel_pose = np.matmul(np.linalg.inv(prev_pose), pose)
            speed1 = np.linalg.norm([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])
            speed2 = 10 * np.linalg.norm([rel_pose[0][3], rel_pose[1][3], rel_pose[2][3]])
            print(speed1, speed2)

        
        prev_pose = pose

        if i == 10:
            break

        # image_buffer = np.fromstring(image.image, np.uint8)
        # cv2_image = cv2.imdecode(image_buffer, -1)

        # examples.append(create_record(cv2_image, speed))


    # write_records(examples, "/mnt/d/waymo/segments/segment_{}.tfrecord".format(i))

extract_segment("/mnt/d/waymo/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord")