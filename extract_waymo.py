import os
import tensorflow as tf
import math
import numpy as np
import itertools
from extract import create_record, write_records
import glob
from multiprocessing import Pool

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

def extract_segment(segment_info):
    i, segment = segment_info
    print("working on segment {}".format(i))
    examples = []
    for j, data in enumerate(tf.data.TFRecordDataset(segment)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        (range_images, camera_projections,
        range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)

        image = frame.images[0]
        speed = np.linalg.norm([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])

        examples.append(create_record(image.image, speed))

    write_records(examples, "/mnt/d/waymo/segments/segment_{}.tfrecord".format(i))

segment_dirs = glob.glob("/mnt/d/waymo/*.tfrecord")

print(len(segment_dirs))

enumerated_dirs = [x for x in enumerate(segment_dirs)]

print(enumerated_dirs[:10])

with Pool(32) as p:
    p.map(extract_segment, enumerated_dirs)

# extract_segment((1, "/mnt/d/waymo/segment-15832924468527961_1564_160_1584_160.tfrecord"))