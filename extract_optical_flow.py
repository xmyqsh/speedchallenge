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

ret, first_frame = cam.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while(True):
    ret, frame = cam.read()
    speed = speeds[current_frame]

    if ret:
        print("Creating {}, speed {}".format(current_frame, speed))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        ret, jpg = cv2.imencode(".jpg", rgb)
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
        print("creating the last frame")
        # we're calculating the flow for frame n with frames n and n+1, so we have to repeat the last frame
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        ret, jpg = cv2.imencode(".jpg", rgb)
        if ret:
            print("appending the last frame")
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(jpg.tostring()),
                'label': _float_feature(float(speed))
            }))
            examples.append(example.SerializeToString())

        break

cam.release()
cv2.destroyAllWindows()

temporal_train_examples = examples[:16320]
temporal_validation_examples = examples[16320:]

write_records(temporal_train_examples, "D:\\speedchallenge\\optical_flows\\temporal\\train.tfrecords")
write_records(temporal_validation_examples, "D:\\speedchallenge\\optical_flows\\temporal\\validation.tfrecords")

random_examples = np.random.permutation(examples)

random_train_examples = random_examples[:16320]
random_validation_examples = random_examples[16320:]

write_records(random_train_examples, "D:\\speedchallenge\\optical_flows\\random\\train.tfrecords")
write_records(random_validation_examples, "D:\\speedchallenge\\optical_flows\\random\\validation.tfrecords")
