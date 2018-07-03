import h5py
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Flatten
from keras.layers.normalization import BatchNormalization

def FireModule(s_1x1, e_1x1, e_3x3, name):
    """
        Fire module for the SqueezeNet model. 
        Implements the expand layer, which has a mix of 1x1 and 3x3 filters, 
        by using two conv layers concatenated in the channel dimension. 
        Returns a callable function
    """
    def layer(x):
        squeeze = Convolution2D(s_1x1, 1, 1, activation='relu', init='he_normal', name=name+'_squeeze')(x)
        squeeze = BatchNormalization(name=name+'_squeeze_bn')(squeeze)
        # Set border_mode to same to pad output of expand_3x3 with zeros.
        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = Convolution2D(e_1x1, 1, 1, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_1x1')(squeeze)
        # expand_1x1 = BatchNormalization(name=name+'_expand_1x1_bn')(expand_1x1)

        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = Convolution2D(e_3x3, 3, 3, border_mode='same', activation='relu', init='he_normal', name=name+'_expand_3x3')(squeeze)
        # expand_3x3 = BatchNormalization(name=name+'_expand_3x3_bn')(expand_3x3)

        expand_merge = merge([expand_1x1, expand_3x3], mode='concat', concat_axis=3, name=name+'_expand_merge')
        return expand_merge
    return layer
    


def SqueezeNet(nb_classes, input_shape=(227, 227, 3)): 
    # Use input shape (227, 227, 3) instead of the (224, 224, 3) shape cited in the paper. 
    # This results in conv1 output shape = (None, 111, 111, 96), same as in the paper. 
    input_image = Input(shape=input_shape)
    conv1 = Convolution2D(96, 7, 7, activation='relu', subsample=(2, 2), init='he_normal', name='conv1')(input_image)
    # conv1 = BatchNormalization(name='conv1_bn')(conv1)
    # maxpool1 output shape = (?, 55, 55, 96)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)

    # fire2 output shape = (?, 55, 55, 128)
    fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
    # fire3 output shape = (?, 55, 55, 128)
    fire3 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire3')(fire2)
    # fire4 output shape = (?, 55, 55, 256)
    fire4 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire4')(fire3)
    # maxpool4 output shape = (?, 27, 27, 256)
    maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool4')(fire4)
    # fire5 output shape = (?, 27, 27, 384)
    fire5 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire5')(maxpool4)
    # fire6 output shape = (?, 27, 27, 384)
    fire6 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire6')(fire5)
    # fire7 output shape = (?, 27, 27, 384)
    fire7 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire7')(fire6)
    # fire8 output shape = (?, 27, 27, 512)
    fire8 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire8')(fire7)
    # maxpool8 output shape = (?, 13, 13, 384). The paper states this output is (13, 13, 512)?
    maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool8')(fire7)
    # fire9 output shape = (?, 13, 13, 512)
    fire9 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire9')(maxpool8)
    # Dropout after fire9 module.
    fire9_dropout = Dropout(p=0.5, name='fire9_dropout')(fire9)

    # conv10 output shape = (?, 13, 13, 6)
    conv10 = Convolution2D(nb_classes, 1, 1, activation='relu', init='he_normal', name='conv10')(fire9_dropout)
    conv10 = BatchNormalization(name='conv10_bn')(conv10)
    # avgpool10, softmax output shape = (?, nb_classes)
    avgpool10 = GlobalAveragePooling2D(name='avgpool10')(conv10)
    # avgpool10 = AveragePooling2D(pool_size=(13, 13), strides=(1, 1), name='avgpool10')(conv10)
    # avgpool10 = Flatten(name='flatten')(avgpool10)
    softmax = Activation('softmax', name='softmax')(avgpool10)

    model = Model(input=input_image, output=[softmax])
    return model