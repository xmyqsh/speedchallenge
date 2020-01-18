import os
import tensorflow as tf
import math
import numpy as np
import itertools
from extract import create_record, write_records
import glob
from multiprocessing import Pool
import cv2
import orientation
import random

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def get_relative_pose(prev_pose, pose):
    rel_pose = np.matmul(np.linalg.inv(prev_pose), pose)
    flu_position = np.array([rel_pose[0][3], rel_pose[1][3], rel_pose[2][3]])
    flu_orientation = np.array(orientation.rot2euler(rel_pose))

    frd_position = flu_position * (1, -1, -1)
    frd_orientation = flu_orientation * (1, -1, -1)

    return frd_position, frd_orientation

def extract_segment(segment):
    frames = []
    poses = []
    speeds = []
    for data in tf.data.TFRecordDataset(segment):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        image = frame.images[0]

        # resize the images and store them in an array
        image_buffer = np.fromstring(image.image, np.uint8)
        frame = cv2.imdecode(image_buffer, -1)
        frame = frame[320:-320]
        frame = cv2.resize(frame, (416, 128), interpolation=cv2.INTER_AREA)
        ret, jpg = cv2.imencode(".jpg", frame)
        if not ret:
            raise Exception("couldn't encode the image")
        frame_bytes = jpg.tostring()
        frames.append(frame_bytes)

        # get the poses as numpy matrixes and store in an array
        pose = np.array(image.pose.transform)
        pose = np.reshape(pose, (4, 4))
        poses.append(pose)

        speed = np.linalg.norm([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])
        speeds.append(speed)

    examples = []
    for i in range(len(frames[:-1])):
        plus_one_position, plus_one_orientation = get_relative_pose(poses[i], poses[i+1])

        example = tf.train.Example(features=tf.train.Features(feature={
            'frame_one': _bytes_feature(frames[i]),
            'frame_two': _bytes_feature(frames[i+1]),
            'frame_three': _bytes_feature(frames[i+1]),
            'frame_four': _bytes_feature(b""),
            'plus_one_position': _float_list_feature(plus_one_position),
            'plus_one_orientation': _float_list_feature(plus_one_orientation),
            'plus_two_position': _float_list_feature([0.0, 0.0, 0.0]),
            'plus_two_orientation': _float_list_feature([0.0, 0.0, 0.0]),
            'plus_three_position': _float_list_feature([0.0, 0.0, 0.0]),
            'plus_three_orientation': _float_list_feature([0.0, 0.0, 0.0]),
            'speed': _float_feature(speeds[i]),
        }))
        examples.append(example.SerializeToString())

    return examples



def write_tfrecord(segments, out):
    with tf.python_io.TFRecordWriter(out) as writer:
        example_buffer = []
        for i, segment in enumerate(segments):
            example_buffer.extend(extract_segment(segment))
            print("extracted {} segments".format(i))

            if len(example_buffer) > 1000000:
                random.shuffle(example_buffer)
                for e in example_buffer:
                    writer.write(e)
                example_buffer = []

                print("\n\nflushed the buffer\n\n")

        random.shuffle(example_buffer)
        for e in example_buffer:
            writer.write(e)

segments = glob.glob("/mnt/Bulk/waymo/*/*.tfrecord")
random.shuffle(segments)

write_tfrecord(segments, "/mnt/Bulk/waymo/monolithic_train.tfrecord")