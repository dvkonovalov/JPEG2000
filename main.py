import time

from PIL import Image
from math import floor
from math import ceil
import numpy as np
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel

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


def transform(image, size, count):
    rows_count = size[0]
    columns_count = size[1]
    image = wavelet_without_loss(image, size)
    k = 1
    while k < count:
        rows_count = ceil(rows_count / 2)
        columns_count = ceil(columns_count / 2)
        quadrant = np.array([[(0, 0, 0) for j in range(columns_count)] for i in range(rows_count)])
        for i in range(rows_count):
            for j in range(columns_count):
                quadrant[i, j] = image[i, j]
        quadrant = wavelet_without_loss(quadrant, (rows_count, columns_count))
        for i in range(rows_count):
            for j in range(columns_count):
                image[i, j] = quadrant[i, j]
        k += 1
    return image



def reverse_transform(image, size, count):
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
        quadrant = wavelet_without_loss_reverse(quadrant, (rows_count, columns_count))

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
        matrix: матрица полученная на 3 шаге"""
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


def get_destribution(matrix, size):
    """
    Получение распределения значений пикселей
    :param matrix: матрица изображения
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: кортеж с словарями распределений
    """
    count_y = {}
    count_cb = {}
    count_cr = {}
    m = 0
    for i in range(size[0]):
        for j in range(size[1]):
            y = matrix[i, j][0]
            cb = matrix[i, j][1]
            cr = matrix[i, j][2]

            if y>m:
                m = y

            if cb>m:
                m = cb

            if cr>m:
                m = cr

            if y not in count_y:
                count_y[y] = 1
            else:
                count_y[y] += 1

            if cb not in count_cb:
                count_cb[cb] = 1
            else:
                count_cb[cb] += 1

            if cr not in count_cr:
                count_cr[cr] = 1
            else:
                count_cr[cr] += 1
    dist_y = {}
    dist_cb = {}
    dist_cr = {}
    pr_y = 0
    pr_cb = 0
    pr_cr = 0
    for i in range(-1000, 1000):
        if i in count_y:
            dist_y[i] = (pr_y, pr_y + count_y[i])
            pr_y = count_y[i] + pr_y

        if i in count_cb:
            dist_cb[i] = (pr_cb, pr_cb + count_cb[i])
            pr_cb = pr_cb + count_cb[i]

        if i in count_cr:
            dist_cr[i] = (pr_cr, pr_cr + count_cr[i])
            pr_cr = pr_cr + count_cr[i]

    return (dist_y, dist_cb, dist_cr)


def mq_coder(matrix, size):
    """
    Арифметическое кодирование (MQ-кодер)
    :param matrix: матрица пикселей после квантования
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: массив со значениями данных после арифметического кодирования,
    кортеж с распределениями из функции get_destribution
    """
    mas_distributions = get_destribution(matrix, size)
    delitel = size[0] * size[1]
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3
    mas = []

    for rounds in range(3):
        string1 = ''
        le = 0
        h = 65535
        bits_to_follow = 0
        distribution = mas_distributions[rounds]
        for i in range(size[0]):
            for j in range(size[1]):
                pixel = matrix[i, j]
                component = pixel[rounds]
                ln = le + (distribution[component][0] * (h - le + 1)) // delitel
                h = le + (distribution[component][1] * (h - le + 1)) // delitel - 1
                le = ln
                if h<le:
                    print(le, h)
                    h = le
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
                    h += (h + 1)

        mas.append(string1)
    return mas, mas_distributions


def mq_coder_revers(mas, size, distrb):
    """
    Арифметическое декодирование (обратный MQ-кодер)
    :param mas: Массив с закодированными последовательностями
    :param size: размеры получаемой матрицы в виде кортежа
    :return: матрица изображения после декодирования
    """
    matrix = np.array([[(0,0,0) for j in range(size[1])] for i in range(size[0])])
    # img = Image.new('RGB', size, 'white')
    # matrix = img.load()
    distribution = distrb
    delitel = size[0] * size[1]
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3

    for rounds in range(3):
        dist_comp = distribution[rounds]
        string = mas[rounds]
        l = 0
        h = 65535
        value = int(string[:16], 2)
        next_pos = 15
        height = 0
        width = 0
        while (next_pos < len(string)):
            freq = ((value - l + 1) * delitel - 1) // (h - l + 1)
            j = 0
            for j in dist_comp:
                if (dist_comp[j][1] <= freq):
                    continue
                else:
                    break
            ln = l + (dist_comp[j][0] * (h - l + 1)) // delitel
            h = l + (dist_comp[j][1] * (h - l + 1)) // delitel - 1
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
                if (next_pos == len(string)):
                    break
                value = value * 2 + int(string[next_pos])
            component_pixel = j
            pixel = matrix[height, width]
            pix = list(pixel)
            pix[rounds] = component_pixel
            pix = tuple(pix)
            matrix[height, width] = pix
            width += 1
            if (width == size[1]):
                height += 1
                width = 0
                if (height == size[0]):
                    break
    return matrix
















def create_file(data, path):
    """
    Функция для записи данных изображения в файл
    :param data: словарь с данными
    :param path: путь куда сохранить файл
    :return: True - успешно выполнено, False - ошибка
    """
    with open(path, 'w') as file:
        """
        Порядок записи:
        1) Размер изображения
        2) Степени ST через пробел
        3) Значение распредления Y
        4) Значение распредления Cb
        5) Значене распредления Cr
        6) Строка значений для Y
        7) Строка значений Cb
        8) Строка значений Cr
        9) Коэффициент квантования
        10) Включено ли повторное вейвлет-преобразование
        """
        wr_record = str(data['size'][0]) + ' ' + str(data['size'][1]) + '\n'
        file.write(wr_record)
        wr_record = str(data['mas_st'][0]) + ' ' + str(data['mas_st'][1]) + ' ' + str(data['mas_st'][2]) + '\n'
        file.write(wr_record)
        for component in data['mas_destribution']:
            for i in component:
                file.write(str(i) + ' ' + str(component[i][1]) + ' ')
            file.write('\n')
        for i in data['mas_values']:
            file.write(i + '\n')
        file.write(str(data['quantize_koef']) + '\n')
        file.write(str(int(data['on_transform'])))


def read_data(path):
    ret_dict = {}
    """
            Порядок чтения:
            1) Размер изображения
            2) Степени ST через пробел
            3) Значение распредления Y
            4) Значение распредления Cb
            5) Значене распредления Cr
            6) Строка значений для Y
            7) Строка значений Cb
            8) Строка значений Cr
            9) Коэффициент квантования
            10) Включено ли повторное вейвлет-преобразование
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


def convert_to_JPEG(path, path_save, quantize_koef=0.1, on_transform=False):
    matrix, size = get_matrix_pixel(path)  # size = (height, width)
    matrix, mas_st = dc_level_shift(matrix, size)
    matrix = convert_image_to_YCbCr(matrix, size)
    if on_transform:
        matrix = transform(matrix, size)
    else:
        matrix = wavelet_without_loss(matrix, size)
    matrix = quantize(matrix, quantize_koef)
    mas_values, mas_destribution = mq_coder(matrix, size)
    rec_dict = {}
    rec_dict['size'] = size
    rec_dict['mas_st'] = mas_st
    rec_dict['quantize_koef'] = quantize_koef
    rec_dict['mas_values'] = mas_values
    rec_dict['mas_destribution'] = mas_destribution
    rec_dict['on_transform'] = on_transform
    create_file(rec_dict, path_save)


def show_image(path):
    data = read_data(path)
    size = data['size']
    matrix = mq_coder_revers(data['mas_values'], size, data['mas_destribution'])
    matrix = reverse_quantize(matrix, size, data['quantize_koef'])
    if data['on_transform']:
        matrix = reverse_transform(matrix, size)
    else:
        matrix = wavelet_without_loss_reverse(matrix, size)
    matrix = convert_image_to_RGB(matrix, size)
    matrix = dc_level_shift_revers(matrix, size, data['mas_st'])
    get_image_from_array(matrix, size)


def convert_image(path, path_save):
    data = read_data(path)
    size = data['size']
    matrix = mq_coder_revers(data['mas_values'], size, data['mas_destribution'])
    matrix = reverse_quantize(matrix, size, data['quantize_koef'])
    if data['on_transform']:
        matrix = reverse_transform(matrix, size)
    else:
        matrix = wavelet_without_loss_reverse(matrix, size)
    matrix = convert_image_to_RGB(matrix, size)
    matrix = dc_level_shift_revers(matrix, size, data['mas_st'])
    save_image(matrix, size, path_save)
#
# convert_to_JPEG('example1.jpg', 'coded1.jpg')
# show_image("coded1.jpg")
# koef = 0.1
matrix, size = get_matrix_pixel('example1.jpg')



# new_matrix = wavelet_without_loss_reverse(wavelet_without_loss(matrix, size), size)
# array = np.zeros((size[0], size[1]))
# file = open('file.txt', 'w')
#
# for i in range(size[0]):
#     for j in range(size[1]):
#         file.write(str(matrix[i, j][0] - new_matrix[i, j][0]) + ' ')
#         array[i, j] = matrix[i, j][0] - new_matrix[i, j][0]
#     file.write('\n')
# file.close()



matrix, mas_st = dc_level_shift(matrix, size)
matrix = convert_image_to_YCbCr(matrix, size)
matrix = transform(matrix, size, 6)
matrix = quantize(matrix, 0.05)
# matrix, dest = mq_coder(matrix, size)
# matrix = mq_coder_revers(matrix, size, dest)
matrix = reverse_quantize(matrix, size, 0.05)
matrix = reverse_transform(matrix, size, 6)
matrix = convert_image_to_RGB(matrix, size)
matrix = dc_level_shift_revers(matrix, size, mas_st)
get_image_from_array(matrix, size)
