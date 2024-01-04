import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import (Input,Add,add,concatenate,Activation,concatenate,
                        Concatenate,Dropout,BatchNormalization,Reshape,Permute,
                        Dense,UpSampling2D,Flatten,Lambda,Activation,Conv2D,
                        DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D,
                        MaxPooling2D,AveragePooling2D,LeakyReLU,Conv2DTranspose)
                        
from keras.regularizers import l2
from keras.utils.layer_utils import get_source_inputs
from keras.utils.data_utils import get_file
from keras.activations import relu
from keras.optimizers import SGD, Adam


weight_decay = 1e-5


def residualDilatedInceptionModule(y, nb_channels, _strides=(1, 1),t="e"):
    if t=="d":
        y = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1),kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        y = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1),kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)


    A1 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(y)
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)
    A1 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), padding='same', use_bias=False)(A1)
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)


    A4 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4),  dilation_rate=4, padding='same', use_bias=False)(y)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)
    A4 = Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides,kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4),  dilation_rate=4, padding='same', use_bias=False)(A4)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)

    if (t=="e"):
        y=concatenate([y,y])
    y=add([A1,A4,y])
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    return y

def PathoNet(input_size = (256,256,3), classes=3, pretrained_weights = None):
    inputs = Input(input_size) 

    block1= Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(inputs)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    block1= Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block1)
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)

    block2= residualDilatedInceptionModule(pool1,32,t="e")
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)

    block3= residualDilatedInceptionModule(pool2,64,t="e")
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)

    block4= residualDilatedInceptionModule(pool3,128,t="e")
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
    drop4 = Dropout(0.1)(pool4)

    block5= residualDilatedInceptionModule(drop4,256,t="e")
    drop5 = Dropout(0.1)(block5)

    up6 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(drop5)),128,t="d")
    merge6 = concatenate([block4,up6], axis = 3)

    up7 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge6)),64,t="d")
    merge7 = concatenate([block3,up7], axis = 3)

    up8 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge7)),32,t="d")
    merge8 = concatenate([block2,up8], axis = 3)

    up9 = residualDilatedInceptionModule((UpSampling2D(size = (2,2))(merge8)),16,t="d")
    merge9 = concatenate([block1,up9], axis = 3)

    block9=Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(merge9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    block9=Conv2D(16, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    block9=Conv2D(8, 3, padding = 'same', kernel_initializer = 'orthogonal',kernel_regularizer= l2(5e-4), use_bias=False)(block9)
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    conv10 = Conv2D(classes, 1, activation = 'relu')(block9)

    model = Model(input = inputs, output = conv10)

    
#     model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

def modelCreator(modelName,inputShape,classes,weights=None):
    classes = 2
    print('-' * 100, '\n', 'classes = ', classes, '\n', '-' * 100)
    if modelName=="PathoNet":
        model=PathoNet(input_size = inputShape, classes=classes,pretrained_weights = weights)
    else:
        raise ValueError('The `model` argument should be either '
                         'PathoNet')
    return model
    