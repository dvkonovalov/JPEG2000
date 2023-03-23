from PIL import Image
from math import floor
from math import ceil
import numpy as np
import copy as cp

from array import *

"""Стандартные матрицы квантования"""
q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

q_c = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 33, 33, 99, 99],
                [99, 99, 99, 99, 33, 33, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])


def wavelet_without_loss(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_1 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_2 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_3 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    result = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    for i in range(rows_count):
        for j in range(1, columns_count, 2):
            if j + 1 != columns_count:
                y = image[i, j][0] - floor((image[i, j - 1][0] + image[i, j + 1][0]) / 2)
                cb = image[i, j][1] - floor((image[i, j - 1][1] + image[i, j + 1][1]) / 2)
                cr = image[i, j][2] - floor((image[i, j - 1][2] + image[i, j + 1][2]) / 2)
            else:
                y = image[i, j][0] - floor((image[i, j - 1][0] + image[i, j - 1][0]) / 2)
                cb = image[i, j][1] - floor((image[i, j - 1][1] + image[i, j - 1][1]) / 2)
                cr = image[i, j][2] - floor((image[i, j - 1][2] + image[i, j - 1][2]) / 2)
            intermediate_1[i, j] = (y, cb, cr)
        for j in range(0, columns_count, 2):
            if j + 1 != columns_count and j != 0:
                y = image[i, j][0] + floor((intermediate_1[i, j - 1][0] + intermediate_1[i, j + 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_1[i, j - 1][1] + intermediate_1[i, j + 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_1[i, j - 1][2] + intermediate_1[i, j + 1][2] + 2) / 4)
            elif j + 1 == columns_count:
                y = image[i, j][0] + floor((intermediate_1[i, j - 1][0] + intermediate_1[i, j - 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_1[i, j - 1][1] + intermediate_1[i, j - 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_1[i, j - 1][2] + intermediate_1[i, j - 1][2] + 2) / 4)
            else:
                y = image[i, j][0] + floor((intermediate_1[i, j + 1][0] + intermediate_1[i, j + 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_1[i, j + 1][1] + intermediate_1[i, j + 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_1[i, j + 1][2] + intermediate_1[i, j + 1][2] + 2) / 4)
            intermediate_1[i, j] = (y, cb, cr)

    first_part = [(0, 0, 0) for _ in range(ceil(columns_count / 2))]
    second_part = [(0, 0, 0) for _ in range(floor(columns_count / 2))]

    for i in range(rows_count):
        for j in range(columns_count):
            if j % 2 == 0:
                first_part[int(j / 2)] = intermediate_1[i, j]
            else:
                second_part[int((j - 1) / 2)] = intermediate_1[i, j]
        array = np.concatenate([first_part, second_part])
        intermediate_2[i] = array

    for j in range(columns_count):
        for i in range(1, rows_count, 2):
            if i + 1 != rows_count:
                y = intermediate_2[i, j][0] - floor((intermediate_2[i + 1, j][0] + intermediate_2[i - 1, j][0]) / 2)
                cb = intermediate_2[i, j][1] - floor((intermediate_2[i + 1, j][1] + intermediate_2[i - 1, j][1]) / 2)
                cr = intermediate_2[i, j][2] - floor((intermediate_2[i + 1, j][2] + intermediate_2[i - 1, j][2]) / 2)
            else:
                y = intermediate_2[i, j][0] - floor((intermediate_2[i - 1, j][0] + intermediate_2[i - 1, j][0]) / 2)
                cb = intermediate_2[i, j][1] - floor((intermediate_2[i - 1, j][1] + intermediate_2[i - 1, j][1]) / 2)
                cr = intermediate_2[i, j][2] - floor((intermediate_2[i - 1, j][2] + intermediate_2[i - 1, j][2]) / 2)
            intermediate_3[i, j] = (y, cb, cr)
        for i in range(0, rows_count, 2):
            if i + 1 != rows_count and i != 0:
                y = intermediate_2[i, j][0] + floor((intermediate_3[i + 1, j][0] + intermediate_3[i - 1, j][0] + 2) / 4)
                cb = intermediate_2[i, j][1] + floor((intermediate_3[i + 1, j][1] + intermediate_3[i - 1, j][1] + 2) / 4)
                cr = intermediate_2[i, j][2] + floor((intermediate_3[i + 1, j][2] + intermediate_3[i - 1, j][2] + 2) / 4)
            elif i + 1 == rows_count:
                y = intermediate_2[i, j][0] + floor((intermediate_3[i - 1, j][0] + intermediate_3[i - 1, j][0] + 2) / 4)
                cb = intermediate_2[i, j][1] + floor((intermediate_3[i - 1, j][1] + intermediate_3[i - 1, j][1] + 2) / 4)
                cr = intermediate_2[i, j][2] + floor((intermediate_3[i - 1, j][2] + intermediate_3[i - 1, j][2] + 2) / 4)
            else:
                y = intermediate_2[i, j][0] + floor((intermediate_3[i + 1, j][0] + intermediate_3[i + 1, j][0] + 2) / 4)
                cb = intermediate_2[i, j][1] + floor((intermediate_3[i + 1, j][1] + intermediate_3[i + 1, j][1] + 2) / 4)
                cr = intermediate_2[i, j][2] + floor((intermediate_3[i + 1, j][2] + intermediate_3[i + 1, j][2] + 2) / 4)
            intermediate_3[i, j] = (y, cb, cr)

    first_part = np.array([(0, 0, 0) for _ in range(ceil(rows_count / 2))])
    second_part = np.array([(0, 0, 0) for _ in range(floor(rows_count / 2))])

    for j in range(columns_count):
        for i in range(rows_count):
            if i % 2 == 0:
                first_part[int(i / 2)] = intermediate_3[i, j]
            else:
                second_part[int((i - 1) / 2)] = intermediate_3[i, j]
        array = np.concatenate([first_part, second_part])
        for i in range(rows_count):
            result[i, j] = array[i]

    return result


def wavelet_without_loss_reverse(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_1 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_2 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_3 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    result = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    border = rows_count / 2
    for j in range(columns_count):
        for i in range(ceil(border)):
            intermediate_1[i * 2, j] = image[i, j]
        for i in range(ceil(border), rows_count):
            intermediate_1[2 * (i - ceil(border)) + 1, j] = image[i, j]

    for j in range(columns_count):
        for i in range(0, rows_count, 2):
            if i + 1 != rows_count and i != 0:
                y = intermediate_1[i, j][0] - floor((intermediate_1[i - 1, j][0] + intermediate_1[i + 1, j][0] + 2) / 4)
                cb = intermediate_1[i, j][1] - floor((intermediate_1[i - 1, j][1] + intermediate_1[i + 1, j][1] + 2) / 4)
                cr = intermediate_1[i, j][2] - floor((intermediate_1[i - 1, j][2] + intermediate_1[i + 1, j][2] + 2) / 4)
            elif i + 1 == rows_count:
                y = intermediate_1[i, j][0] - floor((intermediate_1[i - 1, j][0] + intermediate_1[i - 1, j][0] + 2) / 4)
                cb = intermediate_1[i, j][1] - floor((intermediate_1[i - 1, j][1] + intermediate_1[i - 1, j][1] + 2) / 4)
                cr = intermediate_1[i, j][2] - floor((intermediate_1[i - 1, j][2] + intermediate_1[i - 1, j][2] + 2) / 4)
            else:
                y = intermediate_1[i, j][0] - floor((intermediate_1[i + 1, j][0] + intermediate_1[i + 1, j][0] + 2) / 4)
                cb = intermediate_1[i, j][1] - floor((intermediate_1[i + 1, j][1] + intermediate_1[i + 1, j][1] + 2) / 4)
                cr = intermediate_1[i, j][2] - floor((intermediate_1[i + 1, j][2] + intermediate_1[i + 1, j][2] + 2) / 4)
            intermediate_2[i, j] = (y, cb, cr)
        for i in range(1, rows_count, 2):
            if i + 1 != rows_count:
                y = intermediate_1[i, j][0] + floor((intermediate_2[i - 1, j][0] + intermediate_2[i + 1, j][0]) / 2)
                cb = intermediate_1[i, j][1] + floor((intermediate_2[i - 1, j][1] + intermediate_2[i + 1, j][1]) / 2)
                cr = intermediate_1[i, j][2] + floor((intermediate_2[i - 1, j][2] + intermediate_2[i + 1, j][2]) / 2)
            else:
                y = intermediate_1[i, j][0] + floor((intermediate_2[i - 1, j][0] + intermediate_2[i - 1, j][0]) / 2)
                cb = intermediate_1[i, j][1] + floor((intermediate_2[i - 1, j][1] + intermediate_2[i - 1, j][1]) / 2)
                cr = intermediate_1[i, j][2] + floor((intermediate_2[i - 1, j][2] + intermediate_2[i - 1, j][2]) / 2)
            intermediate_2[i, j] = (y, cb, cr)

    border = columns_count / 2
    for i in range(rows_count):
        for j in range(ceil(border)):
            intermediate_3[i, j * 2] = intermediate_2[i, j]
        for j in range(ceil(border), columns_count):
            intermediate_3[i, 1 + 2 * (j - ceil(border))] = intermediate_2[i, j]

    for i in range(rows_count):
        for j in range(0, columns_count, 2):
            if j != 0 and j + 1 != columns_count:
                y = intermediate_3[i, j][0] - floor((intermediate_3[i, j - 1][0] + intermediate_3[i, j + 1][0] + 2) / 4)
                cb = intermediate_3[i, j][1] - floor((intermediate_3[i, j - 1][1] + intermediate_3[i, j + 1][1] + 2) / 4)
                cr = intermediate_3[i, j][2] - floor((intermediate_3[i, j - 1][2] + intermediate_3[i, j + 1][2] + 2) / 4)
            elif j + 1 == columns_count:
                y = intermediate_3[i, j][0] - floor((intermediate_3[i, j - 1][0] + intermediate_3[i, j - 1][0] + 2) / 4)
                cb = intermediate_3[i, j][1] - floor((intermediate_3[i, j - 1][1] + intermediate_3[i, j - 1][1] + 2) / 4)
                cr = intermediate_3[i, j][2] - floor((intermediate_3[i, j - 1][2] + intermediate_3[i, j - 1][2] + 2) / 4)
            else:
                y = intermediate_3[i, j][0] - floor((intermediate_3[i, j + 1][0] + intermediate_3[i, j + 1][0] + 2) / 4)
                cb = intermediate_3[i, j][1] - floor((intermediate_3[i, j + 1][1] + intermediate_3[i, j + 1][1] + 2) / 4)
                cr = intermediate_3[i, j][2] - floor((intermediate_3[i, j + 1][2] + intermediate_3[i, j + 1][2] + 2) / 4)
            result[i, j] = (y, cb, cr)
        for j in range(1, columns_count, 2):
            if j + 1 != columns_count:
                y = intermediate_3[i, j][0] + floor((result[i, j - 1][0] + result[i, j + 1][0]) / 2)
                cb = intermediate_3[i, j][1] + floor((result[i, j - 1][1] + result[i, j + 1][1]) / 2)
                cr = intermediate_3[i, j][2] + floor((result[i, j - 1][2] + result[i, j + 1][2]) / 2)
            else:
                y = intermediate_3[i, j][0] + floor((result[i, j - 1][0] + result[i, j - 1][0]) / 2)
                cb = intermediate_3[i, j][1] + floor((result[i, j - 1][1] + result[i, j - 1][1]) / 2)
                cr = intermediate_3[i, j][2] + floor((result[i, j - 1][2] + result[i, j - 1][2]) / 2)
            result[i, j] = (y, cb, cr)
    return result


def hl(number, imparity):
    a = abs(number)
    if imparity % 2 == 0:
        if a == 0:
            return 1.115087052456994
        elif a == 1:
            return 0.5912717631142470
        elif a == 2:
            return -0.05754352622849957
        elif a == 3:
            return -0.0912717611424948
        else:
            return 0
    else:
        if a == 0:
            return 0.6029490182363579
        elif a == 1:
            return -0.2668641184428723
        elif a == 2:
            return -0.07822326652898785
        elif a == 3:
            return -0.01686411844287495
        elif a == 4:
            return 0.02674875741080976
        else:
            return 0


def gl(number, imparity):
    a = abs(number)
    if imparity % 2 != 0:
        if a == 0:
            return 0.6029490182363579
        elif a == 1:
            return -0.2668641184428723
        elif a == 2:
            return -0.07822326652898785
        elif a == 3:
            return 0.01686411844287495
        elif a == 4:
            return 0.02674875741080976
        else:
            return 0
    else:
        if a == 0:
            return 1.115087052456994
        elif a == 1:
            return 0.5912717631142470
        elif a == 2:
            return -0.05754352622849957
        elif a == 3:
            return -0.09127176311424948
        elif a == 4:
            return 0


def wavelet_with_loss(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_1 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_2 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_3 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    result = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    for i in range(rows_count):
        for j in range(columns_count):
            imparity = j % 2
            if j >= 4 and j + 4 < columns_count:
                y = sum([image[i, j + k][0] * hl(k, imparity) for k in range(-4, + 4 + 1)])
                cb = sum([image[i, j + k][1] * hl(k, imparity) for k in range(-4, + 4 + 1)])
                cr = sum([image[i, j + k][2] * hl(k, imparity) for k in range(-4, + 4 + 1)])
            elif j < 4:
                y = sum([image[i, abs(j + k)][0] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([image[i, abs(j + k)][1] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([image[i, abs(j + k)][2] * hl(k, imparity) for k in range(-4, 4 + 1)])
            else:
                distance = columns_count - j - 1
                count = 4 - distance
                y = sum([image[i, j + k][0] * hl(k, imparity) for k in range(-4, columns_count - j)])
                y += sum([image[i, columns_count - k - 1][0] * hl(distance + k, imparity) for k in range(1, count + 1)])
                cb = sum([image[i, j + k][1] * hl(k, imparity) for k in range(-4, columns_count - j)])
                cb += sum([image[i, columns_count - k - 1][1] * hl(distance + k, imparity) for k in range(1, count + 1)])
                cr = sum([image[i, j + k][2] * hl(k, imparity) for k in range(-4, columns_count - j)])
                cr += sum([image[i, columns_count - k - 1][2] * hl(distance + k, imparity) for k in range(1, count + 1)])
            intermediate_1[i, j] = (y, cb, cr)

    first_part = [(0, 0, 0) for _ in range(ceil(columns_count / 2))]
    second_part = [(0, 0, 0) for _ in range(floor(columns_count / 2))]
    for i in range(rows_count):
        for j in range(columns_count):
            if j % 2 == 0:
                first_part[int(j / 2)] = intermediate_1[i, j]
            else:
                second_part[int((j - 1) / 2)] = intermediate_1[i, j]
        array = np.concatenate([first_part, second_part])
        intermediate_2[i] = array

    for j in range(columns_count):
        for i in range(rows_count):
            imparity = i % 2
            if i >= 4 and i + 4 < rows_count:
                y = sum([intermediate_2[i + k, j][0] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_2[i + k, j][1] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_2[i + k, j][2] * hl(k, imparity) for k in range(-4, 4 + 1)])
            elif i < 4:
                y = sum([intermediate_2[abs(i + k), j][0] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_2[abs(i + k), j][1] * hl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_2[abs(i + k), j][2] * hl(k, imparity) for k in range(-4, 4 + 1)])
            else:
                distance = rows_count - i - 1
                count = 4 - distance
                y = sum([intermediate_2[i + k, j][0] * hl(k, imparity) for k in range(-4, rows_count - i)])
                y += sum([intermediate_2[rows_count - k - 1, j][0] * hl(distance + k, imparity) for k in range(1, count + 1)])
                cb = sum([intermediate_2[i + k, j][1] * hl(k, imparity) for k in range(-4, rows_count - i)])
                cb += sum([intermediate_2[rows_count - k - 1, j][1] * hl(distance + k, imparity) for k in range(1, count + 1)])
                cr = sum([intermediate_2[i + k, j][2] * hl(k, imparity) for k in range(-4, rows_count - i)])
                cr += sum([intermediate_2[rows_count - k - 1, j][2] * hl(distance + k, imparity) for k in range(1, count + 1)])
            intermediate_3[i, j] = (y, cb, cr)

    first_part = np.array([(0, 0, 0) for _ in range(ceil(rows_count / 2))])
    second_part = np.array([(0, 0, 0) for _ in range(floor(rows_count / 2))])
    for j in range(columns_count):
        for i in range(rows_count):
            if i % 2 == 0:
                first_part[int(i / 2)] = intermediate_3[i, j]
            else:
                second_part[int((i - 1) / 2)] = intermediate_3[i, j]
        array = np.concatenate([first_part, second_part])
        for i in range(rows_count):
            result[i, j] = array[i]

    return result


def wavelet_with_loss_reverse(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_1 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_2 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    intermediate_3 = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    result = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
    border = rows_count / 2
    for j in range(columns_count):
        for i in range(ceil(border)):
            intermediate_1[i * 2, j] = image[i, j]
        for i in range(ceil(border), rows_count):
            intermediate_1[2 * (i - ceil(border)) + 1, j] = image[i, j]

    for j in range(columns_count):
        for i in range(rows_count):
            imparity = i % 2
            if i >= 4 and i + 4 < rows_count:
                y = sum([intermediate_1[i + k, j][0] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_1[i + k, j][1] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_1[i + k, j][2] * gl(k, imparity) for k in range(-4, 4 + 1)])
            elif i < 4:
                y = sum([intermediate_1[abs(i + k), j][0] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_1[abs(i + k), j][1] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_1[abs(i + k), j][2] * gl(k, imparity) for k in range(-4, 4 + 1)])
            else:
                distance = rows_count - i - 1
                count = 4 - distance
                y = sum([intermediate_1[i + k, j][0] * gl(k, imparity) for k in range(-4, rows_count - i)])
                y += sum([intermediate_1[rows_count - k - 1, j][0] * gl(distance + k, imparity) for k in range(1, count + 1)])
                cb = sum([intermediate_1[i + k, j][1] * gl(k, imparity) for k in range(-4, rows_count - i)])
                cb += sum([intermediate_1[rows_count - k - 1, j][1] * gl(distance + k, imparity) for k in range(1, count + 1)])
                cr = sum([intermediate_1[i + k, j][2] * gl(k, imparity) for k in range(-4, rows_count - i)])
                cr += sum([intermediate_1[rows_count - k - 1, j][2] * gl(distance + k, imparity) for k in range(1, count + 1)])
            intermediate_2[i, j] = (y, cb, cr)

    border = columns_count / 2
    for i in range(rows_count):
        for j in range(ceil(border)):
            intermediate_3[i, j * 2] = intermediate_2[i, j]
        for j in range(ceil(border), columns_count):
            intermediate_3[i, 1 + 2 * (j - ceil(border))] = intermediate_2[i, j]

    for i in range(rows_count):
        for j in range(columns_count):
            imparity = j % 2
            if j >= 4 and j + 4 < columns_count:
                y = sum([intermediate_3[i, j + k][0] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_3[i, j + k][1] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_3[i, j + k][2] * gl(k, imparity) for k in range(-4, 4 + 1)])
            elif j < 4:
                y = sum([intermediate_3[i, abs(j + k)][0] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cb = sum([intermediate_3[i, abs(j + k)][1] * gl(k, imparity) for k in range(-4, 4 + 1)])
                cr = sum([intermediate_3[i, abs(j + k)][2] * gl(k, imparity) for k in range(-4, 4 + 1)])
            else:
                distance = columns_count - j - 1
                count = 4 - distance
                y = sum([intermediate_3[i, j + k][0] * gl(k, imparity) for k in range(-4, columns_count - j)])
                y += sum([intermediate_3[i, columns_count - k - 1][0] * gl(distance + k, imparity) for k in range(1, count + 1)])
                cb = sum([intermediate_3[i, j + k][1] * gl(k, imparity) for k in range(-4, columns_count - j)])
                cb += sum([intermediate_3[i, columns_count - k - 1][1] * gl(distance + k, imparity) for k in range(1, count + 1)])
                cr = sum([intermediate_3[i, j + k][2] * gl(k, imparity) for k in range(-4, columns_count - j)])
                cr += sum([intermediate_3[i, columns_count - k - 1][2] * gl(distance + k, imparity) for k in range(1, count + 1)])
            result[i, j] = (y, cb, cr)

    return result


def transform(image, size, count, without_loss=True):
    rows_count = size[0]
    columns_count = size[1]
    if without_loss == True:
        image = wavelet_without_loss(image, size)
    else:
        image = wavelet_with_loss(image, size)
    k = 1
    while k < count:
        rows_count = ceil(rows_count / 2)
        columns_count = ceil(columns_count / 2)
        quadrant = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
        for i in range(rows_count):
            for j in range(columns_count):
                quadrant[i, j] = image[i, j]
        if without_loss == True:
            quadrant = wavelet_without_loss(quadrant, (rows_count, columns_count))
        else:
            quadrant = wavelet_with_loss(quadrant, (rows_count, columns_count))
        for i in range(rows_count):
            for j in range(columns_count):
                image[i, j] = quadrant[i, j]
        k += 1
    return image


def reverse_transform(image, size, count, without_loss=True):
    k = 1
    rows_count, columns_count = size
    while k <= count:
        for i in range(count - k):
            rows_count = ceil(rows_count / 2)
            columns_count = ceil(columns_count / 2)
        quadrant = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
        for i in range(rows_count):
            for j in range(columns_count):
                quadrant[i, j] = image[i, j]
        if without_loss == True:
            quadrant = wavelet_without_loss_reverse(quadrant, (rows_count, columns_count))
        else:
            quadrant = wavelet_with_loss_reverse(quadrant, (rows_count, columns_count))
        for i in range(rows_count):
            for j in range(columns_count):
                image[i, j] = quadrant[i, j]
        k += 1
        rows_count, columns_count = size

    return image



def quantize(matrix, n):
    """n: кэф квантования
       matrix: матрица полученная на 3 шаге"""
    new_matrix = np.array([[(0, 0, 0)] * len(matrix[0])] * len(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (q_y[i % 8][j % 8] * n) < 1:
                k = round(matrix[i, j, 0] / 1)
            else:
                k = round(matrix[i, j, 0] / (q_y[i % 8][j % 8] * n))
            new_matrix[i, j, 0] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (q_c[i % 8][j % 8] * n) < 1:
                k = round(matrix[i][j][1] / 1)
            else:
                k = round(matrix[i][j][1] / (q_c[i % 8][j % 8] * n))
            new_matrix[i, j, 1] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (q_c[i % 8][j % 8] * n) < 1:
                k = round(matrix[i][j][2] / 1)
            else:
                k = round(matrix[i][j][2] / (q_c[i % 8][j % 8] * n))
            new_matrix[i, j, 2] = k

    return new_matrix


def reverse_quantize(matrix, size, n):
    """n: кэф квантования
        matrix: матрица полученная на предыдущем шаге"""
    original_matrix = np.array([[(0, 0, 0)] * size[1]] * size[0])
    for i in range(size[0]):
        for j in range(size[1]):
            if (q_y[i % 8][j % 8] * n) < 1:
                k = round(matrix[i, j][0] * 1)
            else:
                k = round(matrix[i, j][0] * (q_y[i % 8][j % 8] * n))
            original_matrix[i, j][0] = k
    for i in range(size[0]):
        for j in range(size[1]):
            if (q_c[i % 8][j % 8] * n) < 1:
                k = round(matrix[i, j][1] * 1)
            else:
                k = round(matrix[i, j][1] * (q_c[i % 8][j % 8] * n))
            original_matrix[i, j][1] = k
    for i in range(size[0]):
        for j in range(size[1]):
            if (q_c[i % 8][j % 8] * n) < 1:
                k = round(matrix[i, j][2] * 1)
            else:
                k = round(matrix[i, j][2] * (q_c[i % 8][j % 8] * n))
            original_matrix[i, j][2] = k

    return original_matrix


def convert_RGB_to_YCbCr(pixel):
    """
    Перевод пикселя из формата RGB в формат YCbCr
    :param pixel: пиксель RGB в формате кортежа
    :return: пиксель YCbCr в формате кортежа
    """
    y = round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
    cb = round(-0.1687 * pixel[0] - 0.3313 * pixel[1] + 0.5 * pixel[2] + 128)
    cr = round(0.5 * pixel[0] - 0.4187 * pixel[1] - 0.0813 * pixel[2] + 128)
    return (y, cb, cr)


def convert_YCbCr_to_RGB(pixel):
    """
    Перевод пикселя из формата YCbCr в формат RGB
    :param pixel: пиксель YCbCr в формате кортежа
    :return: пиксель RGB в формате кортежа
    """
    r = round(pixel[0] + 1.4 * (pixel[2] - 128))
    g = round(pixel[0] - 0.343 * (pixel[1] - 128) - 0.711 * (pixel[2] - 128))
    b = round(pixel[0] + 1.765 * (pixel[1] - 128))
    return (r, g, b)


def get_image_from_array(massiv, size):
    """
    Функция получения изображения из массива numpy
    :param massiv: мсассив numpy
    :param size: размер изображения
    :return: матрица pillow
    """
    img = Image.new('RGB', size, 'white')
    matrix = img.load()
    for i in range(size[0]):
        for j in range(size[1]):
            matrix[i, j] = tuple(massiv[i, j])
    img.show()
    img.save('test1111.jpg')
    return True


def save_image(massiv, size, path):
    """
    Конвертируем изображение из jpeg2000 в jpg
    :param massiv: массив со значениями пикселей
    :param size: размер изображения
    :return: True - Успешно сохранена, False - ошибка
    """
    try:
        img = Image.new('RGB', size, 'white')
        matrix = img.load()
        for i in range(size[0]):
            for j in range(size[1]):
                matrix[i, j] = tuple(massiv[i, j])
        img.save(path, 'JPEG')
        return True
    except:
        return False


def get_matrix_pixel(path):
    """
    Получение матрицы пикселей и ее размера
    :param path: путь до файла изображения
    :return: матрица пикселей, кортеж размера (height, width)
    """
    img = Image.open(path)
    matrix = img.load()
    (height, width) = img.size
    return matrix, (height, width)


def dc_level_shift(matrix, size):
    """
    Выполнение сдвига яркости изображения (DC level shift)
    :param matrix: матрица пикселей изображения
    :param size: кортеж размеров - (высота, ширина)
    :return: Матрица изображения после сдвига яркости, массив со значениями степеней ST
    """
    st = []
    matr = np.array([[(0, 0, 0)] * size[1]] * size[0])
    for components in range(3):
        summa = 0
        count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                summa += matrix[i, j][components]
                count += 1
        degree = 0
        summa = summa // count
        while (2 ** degree <= summa):
            degree += 1
        if (2 ** degree - summa <= summa - 2 ** (degree - 1)):
            st.append(degree - 1)
        else:
            st.append(degree - 2)
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            mas = list(pixel)
            for color in range(3):
                mas[color] -= 2 ** st[color]
            pixel = tuple(mas)
            matr[i, j] = pixel
    return matr, st


def dc_level_shift_revers(matrix, size, st):
    """
    Выполняет возрат сдвига яркости в изображение
    :param matrix: матрица пикселей изображения со сдвигом яркости
    :param size: кортеж размеров - (высота, ширина)
    :param st: кортеж значений степеней ST для как каждой компоненты - (R, G, B)
    :return: Матрица изображения без сдвига яркости
    """
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            for color in range(3):
                mas = list(pixel)
                mas[color] += 2 ** st[color]
                pixel = tuple(mas)
            matrix[i, j] = pixel
    return matrix


def convert_image_to_YCbCr(matrix, size):
    """
    Переводит матрицу пикселей изображения из RGB в YCbCr
    :param matrix: матрица пикселей изображения
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: матрицу изображения в формате YCbCr
    """
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            pixel = convert_RGB_to_YCbCr(pixel)
            matrix[i, j] = pixel
    return matrix


def convert_image_to_RGB(matrix, size):
    """
    Переводит матрицу пикселей изображения из YCbCr в RGB
    :param matrix: матрица пикселей изображения
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: матрицу изображения в формате RGB
    """
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            pixel = convert_YCbCr_to_RGB(pixel)
            matrix[i, j] = pixel
    return matrix


def get_destribution():
    dest = {}
    pr = 0
    for i in range(-384, 385):
        dest[i] = [pr, pr + 1]
        pr += 1
    return dest


def update_destribution(distribution, element):
    pr = 0
    for i in distribution:
        distribution[i][0] += pr
        distribution[i][1] += pr
        if i == element:
            distribution[i][1] += 1
            pr += 1
    return distribution


def mq_coder(matrix, size):
    """
    Арифметическое кодирование (MQ-кодер)
    :param matrix: матрица пикселей после квантования
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: массив со значениями данных после арифметического кодирования,
    кортеж с распределениями из функции get_destribution
    """
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3
    mas = [[], [], []]
    distribution_original = get_destribution()
    delitel_original = distribution_original[384][1]

    for rounds in range(3):
        for i in range(size[0]):
            string1 = ''
            distribution = cp.deepcopy(distribution_original)
            delitel = delitel_original
            le = 0
            h = 65535
            bits_to_follow = 0
            for j in range(size[1]):
                pixel = matrix[i, j]
                component = pixel[rounds]
                ln = le + (distribution[component][0] * (h - le + 1)) // delitel
                h = le + (distribution[component][1] * (h - le + 1)) // delitel - 1
                le = ln
                while (True):
                    if (h < half):
                        string1 += '0' + '1' * bits_to_follow
                        bits_to_follow = 0
                    elif (le >= half):
                        string1 += '1' + '0' * bits_to_follow
                        bits_to_follow = 0
                        le -= half
                        h -= half
                    elif ((le >= first_qtr) and (h < third_qtr)):
                        bits_to_follow += 1
                        le -= first_qtr
                        h -= first_qtr
                    else:
                        break
                    le += le
                    h += h + 1

                distribution = update_destribution(distribution, component)
                delitel += 1

            mas[rounds].append(string1)
    return mas


def mq_coder_revers(mas_data, size):
    """
    Арифметическое декодирование (обратный MQ-кодер)
    :param mas_data: Массив с закодированными последовательностями
    :param size: размеры получаемой матрицы в виде кортежа
    :return: матрица изображения после декодирования
    """
    matrix = np.array([[(0, 0, 0) for j in range(size[1])] for i in range(size[0])])
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3
    distribution_original = get_destribution()
    delitel_original = distribution_original[384][1]

    for rounds in range(3):
        mas = mas_data[rounds]
        for height in range(size[0]):
            distribution = cp.deepcopy(distribution_original)
            delitel = delitel_original
            string1 = mas[height]
            l = 0
            h = 65535
            value = int(string1[:16], 2)
            next_pos = 15
            width = 0
            while (next_pos < len(string1)):
                freq = ((value - l + 1) * delitel - 1) // (h - l + 1)
                for j in distribution:
                    if (distribution[j][1] <= freq):
                        continue
                    else:
                        break
                ln = l + (distribution[j][0] * (h - l + 1)) // delitel
                h = l + (distribution[j][1] * (h - l + 1)) // delitel - 1
                l = ln
                while (True):
                    if (h < half):
                        pass
                    elif (l >= half):
                        l -= half
                        h -= half
                        value -= half
                    elif ((l >= first_qtr) and (h < third_qtr)):
                        l -= first_qtr
                        h -= first_qtr
                        value -= first_qtr
                    else:
                        break
                    l += l
                    h += h + 1
                    next_pos += 1
                    if (next_pos >= len(string1)):
                        break
                    value = value * 2 + int(string1[next_pos])

                component_pixel = j
                pixel = matrix[height, width]
                pix = list(pixel)
                pix[rounds] = component_pixel
                pix = tuple(pix)
                matrix[height, width] = pix
                width += 1

                distribution = update_destribution(distribution, j)
                delitel += 1
    return matrix


def create_file(data):
    """
    Функция для записи данных изображения в файл
    :param data: словарь с данными
    :return: True - успешно выполнено, False - ошибка
    """
    """
    Порядок записи:
    1) Высота изображения
    2) Ширина изображения
    3) Степень ST 1
    4) Степень ST 2
    5) Степень ST 3
    6) Вайвлет с поерями True или без потерь False
    7) Коэффициент квантования
    8) Строка значений для Y
    9) Строка значений Cb
    10) Строка значений Cr
    """

    file = open('photo.bin', 'wb')

    file.write(chr(data['size'][0]).encode() + b"\n")
    file.write(chr(data['size'][1]).encode() + b"\n")

    file.write(chr(data['mas_st'][0]).encode() + b"\n")
    file.write(chr(data['mas_st'][1]).encode() + b"\n")
    file.write(chr(data['mas_st'][2]).encode() + b"\n")
    file.write(str(int(data['quantize_koef'])).encode() + b"\n")
    file.write(str(data['quantize_koef']).encode() + b"\n")


    for i in range(data['size'][1]):
        splits = [data['mas_values'][0][i][x:x + 8] for x in range(0, len(data['mas_values'][0][i]) - 8, 8)]
        bin_array_in = array('B')
        for split in splits:
            bin_array_in.append(int(split, 2))
        bin_array_in.tofile(file)
        file.write(b"\n")
        file.write(b"\n")


        splits = [data['mas_values'][1][i][x:x + 8] for x in range(0, len(data['mas_values'][1][i]) - 8, 8)]
        bin_array_in = array('B')
        for split in splits:
            bin_array_in.append(int(split, 2))
        bin_array_in.tofile(file)
        file.write(b"\n")
        file.write(b"\n")


        splits = [data['mas_values'][2][i][x:x + 8] for x in range(0, len(data['mas_values'][2][i]) - 8, 8)]
        bin_array_in = array('B')
        for split in splits:
            bin_array_in.append(int(split, 2))
        bin_array_in.tofile(file)
        file.write(b"\n")
        file.write(b"\n")



def read_data(path):
    ret_dict = {}
    """
            Порядок чтения:
            1) Высота изображения
            2) Ширина изображения
            3) Степень ST 1
            4) Степень ST 2
            5) Степень ST 3
            6) Вайвлет с поерями True или без потерь False
            7) Коэффициент квантования
            8) Строка значений для Y
            9) Строка значений Cb
            10) Строка значений Cr
            """
    with open(path, 'r') as file:
        temp = list(map(int, (file.readline()).split()))
        ret_dict['size'] = tuple(temp)
        temp = list(map(int, (file.readline()).split()))
        ret_dict['mas_st'] = temp
        mas_destribution = []
        for k in range(3):
            mas_dest = list(map(int, (file.readline()).split()))
            slovar = {}
            pr = 0
            for i in range(0, len(mas_dest), 2):
                slovar[mas_dest[i]] = (pr, mas_dest[i + 1])
                pr = mas_dest[i + 1]
            mas_destribution.append(slovar)
        ret_dict['mas_destribution'] = mas_destribution
        mas_values = []
        for k in range(3):
            mas_values.append(file.readline()[:-1])
        ret_dict['mas_values'] = mas_values
        ret_dict['quantize_koef'] = float(file.readline())
        flag = file.readline()
        if int(flag) == 1:
            flag = True
        else:
            flag = False
        ret_dict['on_transform'] = flag
    return ret_dict


def convert_to_JPEG(path, quantize_koef=0.1, walvet_with_loss = False):
    matrix, size = get_matrix_pixel(path)  # size = (height, width)
    matrix, mas_st = dc_level_shift(matrix, size)
    matrix = convert_image_to_YCbCr(matrix, size)
    if walvet_with_loss:
        matrix = transform(matrix, size, 6, walvet_with_loss)
    else:
        matrix = transform(matrix, size, 6)
    matrix = quantize(matrix, quantize_koef)
    mas_values = mq_coder(matrix, size)
    rec_dict = {}
    rec_dict['size'] = size
    rec_dict['mas_st'] = mas_st
    rec_dict['quantize_koef'] = quantize_koef
    rec_dict['mas_values'] = mas_values
    rec_dict['walvet_with_loss'] = walvet_with_loss
    create_file(rec_dict)


def show_image(path):
    data = read_data(path)
    size = data['size']
    matrix = mq_coder_revers(data['mas_values'], size)
    matrix = reverse_quantize(matrix, size, data['quantize_koef'])
    matrix = reverse_transform(matrix, size, 6)
    matrix = convert_image_to_RGB(matrix, size)
    matrix = dc_level_shift_revers(matrix, size, data['mas_st'])
    get_image_from_array(matrix, size)


def convert_image(path, path_save):
    data = read_data(path)
    size = data['size']
    matrix = mq_coder_revers(data['mas_values'], size)
    matrix = reverse_quantize(matrix, size, data['quantize_koef'])
    matrix = reverse_transform(matrix, size, 6)
    matrix = convert_image_to_RGB(matrix, size)
    matrix = dc_level_shift_revers(matrix, size, data['mas_st'])
    save_image(matrix, size, path_save)





# koef = 0.1
#
# matrix, size = get_matrix_pixel('example3.jpg')
# matrix, mas_st = dc_level_shift(matrix, size)
# matrix = convert_image_to_YCbCr(matrix, size)
#
# matrix = transform(matrix, size, 6)
# matrix = quantize(matrix, koef)
#
# matrix = mq_coder(matrix, size)
# print('Закодировано')
# rec_dict = {}
# rec_dict['size'] = size
# rec_dict['mas_st'] = mas_st
# rec_dict['quantize_koef'] = koef
# rec_dict['mas_values'] = matrix
# create_file(rec_dict)
# print("Записано")
# matrix = mq_coder_revers(matrix, size)
#
# matrix = reverse_quantize(matrix, size, koef)
# matrix = reverse_transform(matrix, size, 6)
#
# matrix = convert_image_to_RGB(matrix, size)
# matrix = dc_level_shift_revers(matrix, size, mas_st)
# get_image_from_array(matrix, size)
