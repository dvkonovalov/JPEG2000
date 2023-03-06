from PIL import Image
import numpy as np
from math import floor
from math import ceil


def wavelet(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_result_1 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    intermediate_result_2 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    intermediate_result_3 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    result = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])

    for i in range(rows_count):
        for j in range(1, columns_count, 2):
            if j + 1 != columns_count:
                y = image[i, j][0] - floor((image[i, j - 1][0] + image[i, j + 1][0]) / 2)
                cb = image[i, j][1] - floor((image[i, j - 1][1] + image[i, j + 1][1]) / 2)
                cr = image[i, j][2] - floor((image[i, j - 1][2] + image[i, j + 1][2]) / 2)
                intermediate_result_1[i, j] = (y, cb, cr)
            else:
                y = image[i, j][0] - floor((image[i, j - 1][0] + image[i, j - 1][0]) / 2)
                cb = image[i, j][1] - floor((image[i, j - 1][1] + image[i, j - 1][1]) / 2)
                cr = image[i, j][2] - floor((image[i, j - 1][2] + image[i, j - 1][2]) / 2)
                intermediate_result_1[i, j] = (y, cb, cr)
        for j in range(0, columns_count, 2):
            if j + 1 != columns_count and j != 0:
                y = image[i, j][0] + floor((intermediate_result_1[i, j - 1][0] + intermediate_result_1[i, j + 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_result_1[i, j - 1][1] + intermediate_result_1[i, j + 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_result_1[i, j - 1][2] + intermediate_result_1[i, j + 1][2] + 2) / 4)
                intermediate_result_1[i, j] = (y, cb, cr)
            elif j + 1 == columns_count:
                y = image[i, j][0] + floor((intermediate_result_1[i, j - 1][0] + intermediate_result_1[i, j - 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_result_1[i, j - 1][1] + intermediate_result_1[i, j - 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_result_1[i, j - 1][2] + intermediate_result_1[i, j - 1][2] + 2) / 4)
                intermediate_result_1[i, j] = (y, cb, cr)
            else:
                y = image[i, j][0] + floor((intermediate_result_1[i, j + 1][0] + intermediate_result_1[i, j + 1][0] + 2) / 4)
                cb = image[i, j][1] + floor((intermediate_result_1[i, j + 1][1] + intermediate_result_1[i, j + 1][1] + 2) / 4)
                cr = image[i, j][2] + floor((intermediate_result_1[i, j + 1][2] + intermediate_result_1[i, j + 1][2] + 2) / 4)
                intermediate_result_1[i, j] = (y, cb, cr)

    for i in range(rows_count):
        first_part = np.array([(0, 0, 0)] * ceil(columns_count / 2))
        second_part = np.array([(0, 0, 0)] * floor(columns_count / 2))
        for j in range(columns_count):
            if j % 2 == 0:
                first_part[int(j / 2)] = intermediate_result_1[i, j]
            else:
                second_part[int((j - 1) / 2)] = intermediate_result_1[i, j]
        array = np.concatenate([first_part, second_part])
        for j in range(columns_count):
            intermediate_result_2[i, j] = array[j]

    for j in range(columns_count):
        for i in range(1, rows_count, 2):
            if i + 1 != rows_count:
                y = intermediate_result_2[i, j][0] - floor((intermediate_result_2[i - 1, j][0] + intermediate_result_2[i + 1, j][0]) / 2)
                cb = intermediate_result_2[i, j][1] - floor((intermediate_result_2[i - 1, j][1] + intermediate_result_2[i + 1, j][1]) / 2)
                cr = intermediate_result_2[i, j][2] - floor((intermediate_result_2[i - 1, j][2] + intermediate_result_2[i + 1, j][2]) / 2)
                intermediate_result_3[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_2[i, j][0] - floor((intermediate_result_2[i - 1, j][0] + intermediate_result_2[i - 1, j][0]) / 2)
                cb = intermediate_result_2[i, j][1] - floor((intermediate_result_2[i - 1, j][1] + intermediate_result_2[i - 1, j][1]) / 2)
                cr = intermediate_result_2[i, j][2] - floor((intermediate_result_2[i - 1, j][2] + intermediate_result_2[i - 1, j][2]) / 2)
                intermediate_result_3[i, j] = (y, cb, cr)
        for i in range(0, rows_count, 2):
            if i + 1 != rows_count and i != 0:
                y = intermediate_result_2[i, j][0] + floor((intermediate_result_3[i - 1, j][0] + intermediate_result_3[i + 1, j][0] + 2) / 4)
                cb = intermediate_result_2[i, j][1] + floor((intermediate_result_3[i - 1, j][1] + intermediate_result_3[i + 1, j][1] + 2) / 4)
                cr = intermediate_result_2[i, j][2] + floor((intermediate_result_3[i - 1, j][2] + intermediate_result_3[i + 1, j][2] + 2) / 4)
                intermediate_result_3[i, j] = (y, cb, cr)
            elif i + 1 == rows_count:
                y = intermediate_result_2[i, j][0] + floor((intermediate_result_3[i - 1, j][0] + intermediate_result_3[i - 1, j][0] + 2) / 4)
                cb = intermediate_result_2[i, j][1] + floor((intermediate_result_3[i - 1, j][1] + intermediate_result_3[i - 1, j][1] + 2) / 4)
                cr = intermediate_result_2[i, j][2] + floor((intermediate_result_3[i - 1, j][2] + intermediate_result_3[i - 1, j][2] + 2) / 4)
                intermediate_result_3[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_2[i, j][0] + floor((intermediate_result_3[i + 1, j][0] + intermediate_result_3[i + 1, j][0] + 2) / 4)
                cb = intermediate_result_2[i, j][1] + floor((intermediate_result_3[i + 1, j][1] + intermediate_result_3[i + 1, j][1] + 2) / 4)
                cr = intermediate_result_2[i, j][2] + floor((intermediate_result_3[i + 1, j][2] + intermediate_result_3[i + 1, j][2] + 2) / 4)
                intermediate_result_3[i, j] = (y, cb, cr)

    for j in range(columns_count):
        first_part = np.array([(0, 0, 0)] * ceil(rows_count / 2))
        second_part = np.array([(0, 0, 0)] * floor(rows_count / 2))
        for i in range(rows_count):
            if i % 2 == 0:
                first_part[int(i / 2)] = intermediate_result_3[i, j]
            else:
                second_part[int((i - 1) / 2)] = intermediate_result_3[i, j]
        array = np.concatenate([first_part, second_part])
        for i in range(rows_count):
            result[i, j] = array[i]
    return result


def wavelet_reverse(image, size):
    rows_count = size[0]
    columns_count = size[1]
    intermediate_result_1 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    intermediate_result_2 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    intermediate_result_3 = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    result = np.array([[(0, 0, 0)]*columns_count for _ in range(rows_count)])
    for j in range(columns_count):
        for i in range(ceil(rows_count / 2)):
            if i < rows_count / 2:
                intermediate_result_1[2 * i, j] = image[i, j]
            else:
                intermediate_result_1[2 * i - 1, j] = image[i, j]

    for j in range(columns_count):
        for i in range(0, rows_count, 2):
            if i + 1 != rows_count and i != 0:
                y = intermediate_result_1[i, j][0] - floor((intermediate_result_1[i - 1, j][0] + intermediate_result_1[i + 1, j][0] + 2) / 4)
                cb = intermediate_result_1[i, j][1] - floor((intermediate_result_1[i - 1, j][1] + intermediate_result_1[i + 1, j][1] + 2) / 4)
                cr = intermediate_result_1[i, j][2] - floor((intermediate_result_1[i - 1, j][2] + intermediate_result_1[i + 1, j][2] + 2) / 4)
                intermediate_result_2[i, j] = (y, cb, cr)
            elif i + 1 == rows_count:
                y = intermediate_result_1[i, j][0] - floor((intermediate_result_1[i - 1, j][0] + intermediate_result_1[i - 1, j][0] + 2) / 4)
                cb = intermediate_result_1[i, j][1] - floor((intermediate_result_1[i - 1, j][1] + intermediate_result_1[i - 1, j][1] + 2) / 4)
                cr = intermediate_result_1[i, j][2] - floor((intermediate_result_1[i - 1, j][2] + intermediate_result_1[i - 1, j][2] + 2) / 4)
                intermediate_result_2[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_1[i, j][0] - floor((intermediate_result_1[i + 1, j][0] + intermediate_result_1[i + 1, j][0] + 2) / 4)
                cb = intermediate_result_1[i, j][1] - floor((intermediate_result_1[i + 1, j][1] + intermediate_result_1[i + 1, j][1] + 2) / 4)
                cr = intermediate_result_1[i, j][2] - floor((intermediate_result_1[i + 1, j][2] + intermediate_result_1[i + 1, j][2] + 2) / 4)
                intermediate_result_2[i, j] = (y, cb, cr)
        for i in range(1, rows_count, 2):
            if i + 1 != rows_count:
                y = intermediate_result_1[i, j][0] + floor((intermediate_result_2[i - 1, j][0] + intermediate_result_2[i + 1, j][0]) / 2)
                cb = intermediate_result_1[i, j][1] + floor((intermediate_result_2[i - 1, j][1] + intermediate_result_2[i + 1, j][1]) / 2)
                cr = intermediate_result_1[i, j][2] + floor((intermediate_result_2[i - 1, j][2] + intermediate_result_2[i + 1, j][2]) / 2)
                intermediate_result_2[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_1[i, j][0] + floor((intermediate_result_2[i - 1, j][0] + intermediate_result_2[i - 1, j][0]) / 2)
                cb = intermediate_result_1[i, j][1] + floor((intermediate_result_2[i - 1, j][1] + intermediate_result_2[i - 1, j][1]) / 2)
                cr = intermediate_result_1[i, j][2] + floor((intermediate_result_2[i - 1, j][2] + intermediate_result_2[i - 1, j][2]) / 2)
                intermediate_result_2[i, j] = (y, cb, cr)

    for i in range(rows_count):
        for j in range(ceil(columns_count / 2)):
            if j < columns_count / 2:
                intermediate_result_3[i, 2 * j] = intermediate_result_2[i, j]
            else:
                intermediate_result_3[i, 2 * j - 1] = intermediate_result_2[i, j]
    for i in range(rows_count):
        for j in range(0, columns_count, 2):
            if j + 1 != columns_count and j != 0:
                y = intermediate_result_3[i, j][0] - floor((intermediate_result_3[i, j - 1][0] + intermediate_result_3[i, j + 1][0] + 2) / 4)
                cb = intermediate_result_3[i, j][1] - floor((intermediate_result_3[i, j - 1][1] + intermediate_result_3[i, j + 1][1] + 2) / 4)
                cr = intermediate_result_3[i, j][2] - floor((intermediate_result_3[i, j - 1][2] + intermediate_result_3[i, j + 1][2] + 2) / 4)
                result[i, j] = (y, cb, cr)
            elif j + 1 == columns_count:
                y = intermediate_result_3[i, j][0] - floor((intermediate_result_3[i, j - 1][0] + intermediate_result_3[i, j - 1][0] + 2) / 4)
                cb = intermediate_result_3[i, j][1] - floor((intermediate_result_3[i, j - 1][1] + intermediate_result_3[i, j - 1][1] + 2) / 4)
                cr = intermediate_result_3[i, j][2] - floor((intermediate_result_3[i, j - 1][2] + intermediate_result_3[i, j - 1][2] + 2) / 4)
                result[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_3[i, j][0] - floor((intermediate_result_3[i, j + 1][0] + intermediate_result_3[i, j + 1][0] + 2) / 4)
                cb = intermediate_result_3[i, j][1] - floor((intermediate_result_3[i, j + 1][1] + intermediate_result_3[i, j + 1][1] + 2) / 4)
                cr = intermediate_result_3[i, j][2] - floor((intermediate_result_3[i, j + 1][2] + intermediate_result_3[i, j + 1][2] + 2) / 4)
                result[i, j] = (y, cb, cr)
        for j in range(1, columns_count, 2):
            if j + 1 != columns_count:
                y = intermediate_result_3[i, j][0] + floor((result[i, j - 1][0] + result[i, j + 1][0]) / 2)
                cb = intermediate_result_3[i, j][1] + floor((result[i, j - 1][1] + result[i, j + 1][1]) / 2)
                cr = intermediate_result_3[i, j][2] + floor((result[i, j - 1][2] + result[i, j + 1][2]) / 2)
                result[i, j] = (y, cb, cr)
            else:
                y = intermediate_result_3[i, j][0] + floor((result[i, j - 1][0] + result[i, j - 1][0]) / 2)
                cb = intermediate_result_3[i, j][1] + floor((result[i, j - 1][1] + result[i, j - 1][1]) / 2)
                cr = intermediate_result_3[i, j][2] + floor((result[i, j - 1][2] + result[i, j - 1][2]) / 2)
                result[i, j] = (y, cb, cr)
    return result


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


def get_matrix_pixel(path):
    """
    Получение матрицы пикселей и ее размера
    :param path: путь до файла изображения
    :return: матрица пикселей, ширина изображения, высота изображения
    """
    img = Image.open(path)
    matrix = img.load()
    (width, height) = img.size
    return matrix, width, height


def dc_level_shift(matrix, size):
    """
    Выполнение сдвига яркости изображения (DC level shift)
    :param matrix: матрица пикселей изображения
    :param size: кортеж размеров - (высота, ширина)
    :return: Матрица изображения после сдвига яркости, массив со значениями степеней ST
    """
    st = []
    for components in range(3):
        summa = 0
        count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                summa += matrix[i, j][components]
                count += 1
        degree = 0
        summa = summa/count
        while (2**degree<summa):
            degree += 1
        if (2**degree-summa<summa-2**(degree-1)):
            st.append(degree)
        else:
            st.append(degree-1)

    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            for color in range(3):
                pixel[color] -= 2 ** (st[color] - 1)
            matrix[i, j] = pixel
    return matrix


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
                pixel[color] += 2 ** (st[color] - 1)
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

# matrica, width, height = get_matrix_pixel('wood.jpg')
# for i in range(height):
#     for j in range(width):
#         pixel = matrica[j, i]
#         print(pixel)
