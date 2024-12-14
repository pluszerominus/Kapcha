from keras.models import Sequential
from keras.layers import *

# Модель для коротких текстов
def ocr_model_numb(alphabet_len):
    model = Sequential()
    # Первый свёрточный блок
    model.add(Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01), input_shape=(280, 70, 1)))
    model.add(MaxPooling2D((2, 2))) # 140x35
    # Второй свёрточный блок
    model.add(Conv2D(128, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2))) # 140x17
    # Третий свёрточный блок
    model.add(Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2))) # 70x8
    model.add(BatchNormalization())
    # Четвёртый свёрточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2))) # 35x4
    # Пятый свёрточный блок
    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2))) # 35x2
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))

    model.add(Reshape((70, 512)))

    # Два слоя LSTM
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))

    # Слой для вывода списка символов считанных с изображения в виде вектора
    model.add(Dense(alphabet_len + 1, activation='softmax')) # +1 для ctc

    return model


