from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.utils import *

from Models.STN import SpatialTransformer

# Модель для STN
def loc_net(input_shape):
    loc_input = Input(input_shape)

    loc_conv_1 = Conv2D(16, (5, 5), padding='same', activation='relu')(loc_input)
    loc_conv_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(loc_conv_1)

    loc_fla = Flatten()(loc_conv_2)
    loc_fc_1 = Dense(64, activation='relu')(loc_fla)
    loc_fc_2 = Dense(6)(loc_fc_1)

    output = Model(inputs=loc_input, outputs=loc_fc_2)

    return output

# Модель для OCR
def CRNN_STN():

    inputShape = Input((width, height, 1))
    # Первый свёрточный блок
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputShape)
    batchnorm_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_1)
    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_2)
    batchnorm_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(batchnorm_3)

    # Второй свёрточный блок
    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_3)
    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_4)
    batchnorm_5 = BatchNormalization()(conv_5)
    pool_5 = MaxPooling2D(pool_size=(2, 2))(batchnorm_5)

    # Третий свёрточный блок
    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_5)
    conv_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv_6)
    batchnorm_7 = BatchNormalization()(conv_7)

    bn_shape = batchnorm_7.shape
    '''=============== STN ==============='''
    stn_input_shape = batchnorm_7.shape
    loc_input_shape = (stn_input_shape[1], stn_input_shape[2], stn_input_shape[3])
    stn = SpatialTransformer(localization_net=loc_net(loc_input_shape),
                             output_size=(loc_input_shape[0], loc_input_shape[1]))(batchnorm_7)
    '''=============== STN ==============='''

    x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(stn)

    fc_1 = Dense(128, activation='relu')(x_reshape)

    # Рекуррентный блок
    rnn_1 = Bidirectional(LSTM(256, return_sequences=True))(fc_1)
    rnn_2 = Bidirectional(LSTM(256, return_sequences=True))(rnn_1)

    drop_1 = Dropout(0.25)(rnn_2)

    fc_2 = Dense(label_classes, kernel_initializer='he_normal', activation='softmax')(drop_1)

    model = Model(inputs=inputShape, outputs=fc_2)

    return model
