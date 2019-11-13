import tensorflow as tf
import imageio
import loader

tf.enable_eager_execution()

training_dataset = loader.load_tfrecord("D:\\commaai\\monolithic_train.tfrecord", True)

for frames, positions, orienations, speeds in training_dataset:
    for i, f in enumerate(frames[:20]):
        imageio.imsave("test/frame{}.jpg".format(i), f)
    for p in positions[:20]:
        print(p)
    break
