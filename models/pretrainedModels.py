import keras.applications

from keras import backend as k
from keras.layers import Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model


class ipLayer(object):

    def _layer_config(self):
        width, height, channels = self.input_shape
        if k.image_data_format() == 'channels_first'
            return(channels, height, width)
        else:
            return(width, height, channels)

class VGG16layer(ipLayer):
    input_shape = (224, 224, 3)

    def __init__(self, num_classes):
        base = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape())
        x = Flatten(name='flatten')(base.output)
        x = Dense()