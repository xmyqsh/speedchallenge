import cv2
import tensorflow as tf
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_records(examples, path):
    writer = tf.python_io.TFRecordWriter(path)
    for e in examples:
        writer.write(e)

def create_record(image_bytes, speed):
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

            ret, jpg = cv2.imencode(".jpg", frame)
            if ret:
                examples.append(create_record(jpg.tostring(), speed))
            else:
                break

            current_frame += 1
        else:
            break

    return examples

def main():
    cam = cv2.VideoCapture("data\\train.mp4")
    speeds = open("data\\train.txt", 'r').readlines()
    speeds = list(map(lambda x: float(x), speeds))

    examples = create_records(cam, speeds)

    cam.release()
    cv2.destroyAllWindows()


    temporal_train_examples = examples[:16320]
    temporal_validation_examples = examples[16320:]
    
    temporal_train_examples_evens = temporal_train_examples[::2]
    temporal_validation_examples_evens = temporal_validation_examples[::2]

    write_records(temporal_train_examples_evens, "D:\\speedchallenge\\temporal\\train_evens.tfrecord")
    write_records(temporal_validation_examples_evens, "D:\\speedchallenge\\temporal\\validation_evens.tfrecord")

    temporal_train_examples_odds = temporal_train_examples[1::2]
    temporal_validation_examples_odds = temporal_validation_examples[1::2]

    write_records(temporal_train_examples_odds, "D:\\speedchallenge\\temporal\\train_odds.tfrecord")
    write_records(temporal_validation_examples_odds, "D:\\speedchallenge\\temporal\\validation_odds.tfrecord")


if __name__ == '__main__':
    main()
