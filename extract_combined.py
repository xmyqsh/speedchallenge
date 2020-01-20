import tensorflow as tf
import glob
import random

import extract_commaai_2k19
import extract_waymo

def write_tfrecord(dirs, out):
    with tf.python_io.TFRecordWriter(out) as writer:
        example_buffer = []
        for i, dir in enumerate(dirs):
            if "waymo" in dir:
                segment = extract_waymo.extract_segment(dir)
            else:
                segment = extract_commaai_2k19.extract_segment(dir)
            example_buffer.extend(segment)
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


waymo_segments = glob.glob("/mnt/Bulk/waymo/*/*.tfrecord")
commaai_segments = glob.glob("/mnt/Bulk/commaai/comma2k19/*/*/*")

segments = []
segments.extend(waymo_segments)
segments.extend(commaai_segments)

random.shuffle(segments)

write_tfrecord(segments, "/mnt/Bulk/monolithic_combined.tfrecord")
