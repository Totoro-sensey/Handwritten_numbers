import cv2
import numpy as np
from os import path
import glob

from config import IMAGES_TRAIN, LABELS_TRAIN


def get_letter_list():
    letter_list = []
    k = 0
    for imageName, textName in zip(sorted(glob.glob(IMAGES_TRAIN)), sorted(glob.glob(LABELS_TRAIN))):

        image_file = (imageName)
        fr = open(textName)
        lines = fr.readlines()
        line_nums = len(lines)

        if line_nums > 1:
            x_mat = []
            contours = np.zeros((line_nums, 4))
            for i in range(line_nums):
                line = lines[i]
                item_mat = line.split(',')
                x_mat.append(item_mat[0:4])  # Получить 4 функции
            fr.close()
            for i in range(line_nums):
                contours[i] = x_mat[i]

            # сортировка по х
            contours = sorted(contours, key=lambda x: x[0], reverse=False)
            print(contours)
            contours_y = contours[0][1]
            contour = []

            # проверка повторения координат
            i = 0
            while i < len(contours) - 1:
                if contours[i][0] >= contours[i + 1][0] - 1:
                    del contours[i]
                i += 1
            # сортировка по у
            i = 0
            for cont in contours:
                if cont[1] > contours_y + 5 or cont[1] < contours_y - 5:
                    contour.append(cont)
                    del contour[i]
                i += 1
                contour.append(cont)

            letters = []
            out_size = 28

            # преобразование изображения
            img = cv2.imread(image_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            img_erode = cv2.erode(thresh, np.ones((1, 2), np.uint8), iterations=1)
            output = img_erode.copy()

            for contour in contours:
                x = int(contour[0] - 8)
                y = int(contour[1] - 8)
                w = int(contour[2])
                h = int(contour[3])
                cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

                letter_crop = img_erode[y:y + h, x:x + w]
                size_max = max(w, h)
                letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

                if w > h:
                    y_pos = size_max // 2 - h // 2
                    letter_square[y_pos:y_pos + h, 0:w] = letter_crop
                elif w < h:
                    x_pos = size_max // 2 - w // 2
                    letter_square[0:h, x_pos:x_pos + w] = letter_crop
                else:
                    letter_square = letter_crop

                letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
            k = k + 1
            num = str(k)
            cv2.imwrite('/content/output_image/' + num + 'output.png', output)

            letter_list.append(letters)

    return letter_list
