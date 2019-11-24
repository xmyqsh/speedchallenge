import cv2
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def write_records(examples, path):
    writer = tf.python_io.TFRecordWriter(path)
    for e in examples:
        writer.write(e)

def create_record(image, speed):
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    ret, jpg = cv2.imencode(".jpg", image)
    if not ret:
        raise Exception("couldn't encode the image")

    image_bytes = jpg.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_bytes),
        'label': _float_feature(speed)
    }))

    return example.SerializeToString()

def create_records(cam, speeds):
    current_frame = 0
    examples = []

    while(True):
        ret, frame = cam.read()

        if ret:
            speed = speeds[current_frame]

            examples.append(create_record(frame, speed))

            current_frame += 1
        else:
            break

    return examples

def main():
    cam = cv2.VideoCapture("data\\train.mp4")
    frame_velocities = open("data\\train.txt", 'r').readlines()
    frame_velocities = list(map(lambda x: float(x), frame_velocities))

    frames = []

    while(True):
        ret, frame = cam.read()

        if ret:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            ret, jpg = cv2.imencode(".jpg", frame)
            if not ret:
                raise Exception("couldn't encode the image")

            frame_bytes = jpg.tostring()
            frames.append(frame_bytes)

        else:
            break


    previous_frame = frames[0]

    examples = []
    for frame, speed in zip(frames[1:], frame_velocities[1:]):
        example = tf.train.Example(features=tf.train.Features(feature={
            'frame_one': _bytes_feature(previous_frame),
            'frame_two': _bytes_feature(frame),
            'position': _float_list_feature([0.0, 0.0, 0.0]),
            'orientation': _float_list_feature([0.0, 0.0, 0.0]),
            'reverse_position': _float_list_feature([0.0, 0.0, 0.0]),
            'reverse_orientation': _float_list_feature([0.0, 0.0, 0.0]),
            'speed': _float_feature(speed),
        }))
        examples.append(example.SerializeToString())

        previous_frame = frame

    with tf.python_io.TFRecordWriter("D:\\speedchallenge\\monolithic_test.tfrecord") as writer:
        for e in examples:
            writer.write(e)



if __name__ == '__main__':
    main()
