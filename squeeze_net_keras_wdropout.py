#Adapated from https://github.com/rcmalli/keras-squeezenet/blob/master/keras_squeezenet/squeezenet.py

from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings, Flatten, Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils

# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, Flatten, Dense
# from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
# from tensorflow.keras.models import Model


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

def add_weight_decay(model, alpha):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(alpha)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(tf.keras.regularizers.l2(alpha)(layer.bias))

# Modular function for Fire Node
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.
def SqueezeNetBinaryKeras(input_shape=None, drop_p = 0):
    """Instantiates the SqueezeNet architecture.
    """
    img_input = Input(shape=input_shape)

    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = Dropout(drop_p, name='drop_added_1')(x)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = Dropout(drop_p, name='drop_added_2')(x)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = Dropout(drop_p, name='drop_added_3')(x)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = Dropout(drop_p, name='drop_added_4')(x)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    x = Convolution2D(128, (1, 1), padding='valid', name='conv10')(x)
    x = Flatten()(x)
    x = Dropout(0.3, name='drop9')(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dense(1, activation = 'sigmoid', name='loss')(x)

    model = Model(img_input, x, name='squeezenet')

    return model

#First test as is
#Then see if including a bottleneck feature + LR layer affects things too badly
#They help!!!
