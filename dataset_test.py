import tensorflow as tf
import imageio
import loader

tf.enable_eager_execution()

training_dataset = loader.load_tfrecord("D:\\commaai\\monolithic_train.tfrecord", 20, True)

for batch in training_dataset:
    # print(batch[1]['pose'])
    for p in batch[1]['pose']:
        print(p)
    # for i, f in enumerate(frames[:20]):
    #     imageio.imsave("test/frame{}.jpg".format(i), f)
    # for p in positions[:20]:
    #     print(p)
    break
