# получение заданных маркеров цифр
import glob
import numpy as np

from config import LABELS_TRAIN


def get_labels():
    labels = []
    for imageLabel in sorted(glob.glob(LABELS_TRAIN)):
        fr = open(imageLabel)
        lines = fr.readlines()
        line_nums = len(lines)
        if line_nums > 1:

            x_mat = []
            contours = np.zeros((line_nums, 5))
            for i in range(line_nums):
                line = lines[i]
                item_mat = line.split(' ')
                x_mat.append(item_mat[0:5])  # Получить 4 функции
            fr.close()

            for i in range(line_nums):
                contours[i] = x_mat[i]
            contours = sorted(contours, key=lambda x: x[0], reverse=False)

            contours_y = contours[0][2]
            contour = []
            i = 0
            for cont in contours:
                if cont[1] > contours_y + 5 or cont[1] < contours_y - 5:
                    contour.append(cont)
                    del contour[i]
                i += 1
                contour.append(cont)

            lab_cont = np.zeros((len(contour), 1))
            for i in range(len(contour)):
                lab_cont[i] = contour[i][0]
            labels.append(lab_cont)
