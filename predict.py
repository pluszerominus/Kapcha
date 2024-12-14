import csv
import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sys import argv
import cv2
from PIL import Image

from src.Models import OCR_short, OCR_long

# Чтение входных параметров
if len(argv) != 1:
    file_path = argv[1]
else:
    file_path = argv
# Путь до используемых моделей
dirname = os.path.dirname(__file__)
short_model_path = os.path.join(dirname, "src", "Models", 'weights', 'OCR_short.keras')
long_model_path = os.path.join(dirname, "src", "Models", 'weights', 'OCR_long.keras')

'''========= short text parameters ========='''
# Список символов
alphabet_short = ['0', '1', '2', '3', '4', '5',
            '6', '7', '8', '9', 'a', 'b',
            'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
alphabet_short_len = len(alphabet_short) + 1

# Словари для кодировки символов
char_label_short = {char:i+1 for i, char in enumerate(alphabet_short)}
label_char_short = {label: char for char, label in char_label_short.items()}

# Инициализация модели для коротких текстов
short_text_model = OCR_short.ocr_model_numb(alphabet_short_len)
short_text_model.load_weights(short_model_path)

'''========= long text parameters ========='''
# Список символов для длинных текстов
alphabet_long = [' ', 'a', 'b', 'c', 'd', 'e',
                 'f', 'g', 'h', 'i', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w',
                 'x', 'y', 'z', 'а', 'б', 'в',
                 'г', 'д', 'е', 'ж', 'з', 'и',
                 'й', 'к', 'л', 'м', 'н', 'о',
                 'п', 'р', 'с', 'т', 'у', 'ф',
                 'х', 'ц', 'ч', 'ш', 'щ', 'ъ',
                 'ы', 'ь', 'э', 'ю', 'я']
alphabet_long_len = len(alphabet_long) + 1

# Словарь для кодировки символов
char_label_long = {char:i for i, char in enumerate(alphabet_long)}
label_char_long = {label: char for char, label in char_label_long.items()}

# Инициализация модели для длинных текстов
long_text_model = OCR_long.ocr_model_text(alphabet_long_len)
long_text_model.load_weights(long_model_path)

'''========= Main part ========='''

# Функция для проверки корректности передаваемого пути
def check_path():
    try:
        img_names = os.listdir(file_path)
    except:
        print("Указан неверный путь")
        return []
    return img_names

# Функция для чтения и обработки изображений
def read_img(img_path):
    img = Image.open(img_path).convert("L")
    img = np.array(img)
    img_type = ""

    # Обработка для изображений с коротким текстом
    if img.shape == (70,280):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = (img / 127.5) - 1
        img_type = "short"
    # Обработка для изображений с длинным текстом
    else:
        img = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    21, 3)
        img = img/255
        img_type = "long"
    return img, img_type

# Функция для декодировки предсказаний модели
def ctc_decode(code, label_char, alphabet_len):
    predict = np.argmax(code, axis=-1)[0]
    text = []
    previos = None
    # Цикл для удаления соседних дубликатов
    for symb in predict:
        if symb != previos:
            previos = symb
            text.append(symb)
    # Декодировка предложения
    text = [label_char[s] for s in text if s != alphabet_len]
    return "".join(text)

# Функция для загрузки результатов в csv
def save_to_csv(text):
    with open("result.csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerow(["img_name", "model_predict"])
        write.writerows(text)

# Основная функция
def main():
    img_name = check_path()
    if len(img_name) == 0:
        return 0

    # Обход всех файлов
    text_list = []
    for name in img_name:
        # Путь до изображения
        img_path = f"{file_path}/{name}"
        try:
            img, img_type = read_img(img_path)
            # Распознавание изображений с коротким текстом
            if img_type == "short":
                code = short_text_model.predict(np.array([img]))
                text = ctc_decode(code, label_char_short, alphabet_short_len)
            # Распознавание изображений с длинным текстом
            else:
                code = long_text_model.predict(np.array([img]))
                text = ctc_decode(code, label_char_long, alphabet_long_len - 1)
            text_list.append([name, text])
            print(f"Текст на изображении '{name}': '{text}'")
        except:
            print(f"Неверный формат файла : {name}")
    # Сохранение результатов в csv
    save_to_csv(text_list)

main()