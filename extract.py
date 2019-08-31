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


cam = cv2.VideoCapture("data\\train.mp4")
speeds = open("data\\train.txt", 'r').readlines()


current_frame = 0
examples = []

while(True):
    ret, frame = cam.read()

    if ret:
        speed = speeds[current_frame]
        print("Creating {}, speed {}".format(current_frame, speed))

        ret, jpg = cv2.imencode(".jpg", frame)
        if ret:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(jpg.tostring()),
                'label': _float_feature(float(speed))
            }))
            examples.append(example.SerializeToString())
        else:
            break

        current_frame += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()


temporal_train_examples = examples[:16320]
temporal_validation_examples = examples[16320:]

write_records(temporal_train_examples, "D:\\speedchallenge\\temporal\\train.tfrecords")
write_records(temporal_validation_examples, "D:\\speedchallenge\\temporal\\validation.tfrecords")

random_examples = np.random.permutation(examples)

random_train_examples = random_examples[:16320]
random_validation_examples = random_examples[16320:]

write_records(random_train_examples, "D:\\speedchallenge\\random\\train.tfrecords")
write_records(random_validation_examples, "D:\\speedchallenge\\random\\validation.tfrecords")
