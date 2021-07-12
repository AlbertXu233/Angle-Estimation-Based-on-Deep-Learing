from tensorflow.keras import Model,layers
from tensorflow.keras.applications import VGG19,VGG16
import tensorflow as tf
from hg_blocks import *
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.set_visible_devices(physical_devices[0],'GPU')
def simpleModel():
    conv_model = VGG19(include_top=False,
              weights='imagenet',
              input_tensor=None,
              input_shape=[650,400,3],
              pooling=None,
              classes=1000)
    conv_model.__setattr__("trainable",False)
    input = conv_model.input
    # x = conv_model.layers[15].output
    x = conv_model.output
    x = layers.Conv2D(256,1,activation='relu')(x)
    x = layers.Conv2D(128,1,activation='relu')(x)
    x = layers.Conv2D(64,1,activation='relu')(x)
    x = layers.Conv2D(6,1)(x)
    x = layers.ReLU()(x)
    # x = layers.Softmax([1,2])(x)
    #回归方案
    # x = layers.Flatten()(x)
    # x = layers.Dense(144)(x)
    # x = layers.Dense(72)(x)
    # x = layers.Dense(36)(x)
    # x = layers.Dense(12)(x)
    # x = tf.reshape(x,[6,2])
    regress_model = Model(input,x)
    adam = tf.keras.optimizers.Adam(0.0001)
    regress_model.compile(adam, loss='mae')
    return regress_model
def attention_loss(y_true,y_predict):
    return tf.abs(y_true-y_predict)*(y_true)