from keras.models import Sequential
from keras.layers import *

# Функция для длинных текстов
def ocr_model_text(alphabet_len):
    model = Sequential()
    # Первый свёрточный блок
    model.add(Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01), input_shape=(81,248, 1)))
    model.add(MaxPooling2D((2, 2))) # 40х124
    # Второй свёрточный блок
    model.add(Conv2D(128, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 1))) # 20х124
    # Третий свёрточный блок
    model.add(Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2))) # 10х62
    model.add(BatchNormalization())
    # Четвёртый свёрточный блок
    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2))) # 5х31
    # Пятый свёрточный блок
    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 1))) # 2х31
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))

    model.add(Reshape((62, 512)))

    # Два слоя LSTM
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))

    # Слой для вывода списка символов считанных с изображения в виде вектора
    model.add(Dense(alphabet_len, activation='softmax')) # +1 для ctc

    return model