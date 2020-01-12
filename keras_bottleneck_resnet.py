import tensorflow as tf

def res_bottleneck_block(input, num_channels, stride=1):
    if not stride == 1 or not input.shape[-1] == num_channels * 4:
        shortcut = tf.keras.layers.Conv2D(num_channels * 4, 1, strides=stride, padding='same')(input)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input

    # Residual
    res = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(input)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.relu)(res)
    res = tf.keras.layers.Conv2D(num_channels, 3, strides=stride, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.relu)(res)
    res = tf.keras.layers.Conv2D(num_channels * 4, 1, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)

    # Merge
    out = tf.keras.layers.add([res, shortcut])
    out = tf.keras.layers.Activation(tf.nn.relu)(out)

    return out

def resnet50_encoder(image):
    encoder_filters = [64, 64, 128, 256, 512]
    stride = 2

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 7, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.relu)(conv1)
    conv1 = tf.keras.layers.MaxPool2D(3, 2, 'same')(conv1)

    conv2 = res_bottleneck_block(conv1, encoder_filters[1])
    conv2 = res_bottleneck_block(conv2, encoder_filters[1])
    conv2 = res_bottleneck_block(conv2, encoder_filters[1])

    conv3 = res_bottleneck_block(conv2, encoder_filters[2], stride)
    conv3 = res_bottleneck_block(conv3, encoder_filters[2])
    conv3 = res_bottleneck_block(conv3, encoder_filters[2])
    conv3 = res_bottleneck_block(conv3, encoder_filters[2])

    conv4 = res_bottleneck_block(conv3, encoder_filters[3], stride)
    conv4 = res_bottleneck_block(conv4, encoder_filters[3])
    conv4 = res_bottleneck_block(conv4, encoder_filters[3])
    conv4 = res_bottleneck_block(conv4, encoder_filters[3])
    conv4 = res_bottleneck_block(conv4, encoder_filters[3])
    conv4 = res_bottleneck_block(conv4, encoder_filters[3])

    conv5 = res_bottleneck_block(conv4, encoder_filters[4], stride)
    conv5 = res_bottleneck_block(conv5, encoder_filters[4])
    conv5 = res_bottleneck_block(conv5, encoder_filters[4])

    return conv5
