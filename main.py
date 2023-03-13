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
    for components in range(3):
        summa = 0
        count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                summa += matrix[i, j][components]
                count += 1
        degree = 0
        summa = summa / count
        while (2 ** degree < summa):
            degree += 1
        if (2 ** degree - summa < summa - 2 ** (degree - 1)):
            st.append(degree)
        else:
            st.append(degree - 1)

    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            for color in range(3):
                pixel[color] -= 2 ** (st[color] - 1)
            matrix[i, j] = pixel
    return matrix, st


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


def get_destribution(matrix, size):
    """
    Получение распределения значений пикселей
    :param matrix: матрица изображения
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: кортеж с словарями распределений
    """
    distribution_y = []
    distribution_cb = []
    distribution_cr = []
    for i in range(256):
        distribution_y.append([i, 0, 0])
        distribution_cb.append([i, 0, 0])
        distribution_cr.append([i, 0, 0])

    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrix[i, j]
            distribution_y[pixel[0]][1] += 1
            distribution_cb[pixel[1]][1] += 1
            distribution_cr[pixel[2]][1] += 1
    # Сортируем полученные массивы распределений
    distribution_y.sort(key=lambda x: x[1], reverse=True)
    distribution_cb.sort(key=lambda x: x[1], reverse=True)
    distribution_cb.sort(key=lambda x: x[1], reverse=True)

    distribution_y[0][2] = distribution_y[0][1]
    distribution_cb[0][2] = distribution_cb[0][1]
    distribution_cr[0][2] = distribution_cr[0][1]

    for i in range(1, 256):
        distribution_y[i][2] = distribution_y[i - 1][2] + distribution_y[i][1]
        distribution_cb[i][2] = distribution_cb[i - 1][2] + distribution_cb[i][1]
        distribution_cr[i][2] = distribution_cr[i - 1][2] + distribution_cr[i][1]

    # переделываем массивы в словари для удобства использования в дальнейшем
    dist_y = {}
    dist_cb = {}
    dist_cr = {}
    pr = 0
    for element in distribution_y:
        dist_y[element[0]] = (pr, element[2])
        pr = element[2]
    pr = 0
    for element in distribution_cb:
        dist_cb[element[0]] = (pr, element[2])
        pr = element[2]
    pr = 0
    for element in distribution_cr:
        dist_cr[element[0]] = (pr, element[2])
        pr = element[2]
    return (dist_y, dist_cb, dist_cr)


def mq_coder(matrix, size):
    """
    Арифметическое кодирование (MQ-кодер)
    :param matrix: матрица пикселей после квантования
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: массив со значениями данных после арифметического кодирования,
    кортеж с распределениями из функции get_destribution
    """
    distribution = get_destribution(matrix, size)
    delitel = size[0] * size[1]
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3
    mas = []

    for rounds in range(3):
        string = ''
        le = 0
        h = 65535
        bits_to_follow = 0
        for i in range(size[0]):
            for j in range(size[1]):
                pixel = matrix[i, j]
                component = pixel[rounds]
                ln = le + (distribution[rounds][component][0] * (h - le + 1)) // delitel
                h = le + (distribution[rounds][component][1] * (h - le + 1)) // delitel - 1
                le = ln
                #фиксим иногда вылет исключения
                if (le > h):
                    h = le
                while (True):
                    if (h < half):
                        string += '0' + '1' * bits_to_follow
                        bits_to_follow = 0
                    elif (le >= half):
                        string += '1' + '0' * bits_to_follow
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

        mas.append(string)
    return mas, distribution


def mq_coder_revers(mas, size, distrb):
    """
    Арифметическое декодирование (обратный MQ-кодер)
    :param mas: Массив с закодированными последовательностями
    :param size: размеры получаемой матрицы в виде кортежа
    :return: матрица изображения после декодирования
    """
    img = Image.new('RGB', size, 'white')
    matrix = img.load()
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
    img.show()
    return matrix


def create_file(data, path):
    """
    Функция для записи данных изображения в файл
    :param data: словарь с данными
    :param path: путь куда сохранить файл
    :return: True - успешно выполнено, False - ошибка
    """
    with open('file.jpeg2000') as file:
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
        """
        wr_record = str(data['size'][0]) + ' ' + str(data['size'][1])
        file.write(wr_record)
        wr_record = str(data['mas_st'][0]) + ' ' + str(data['mas_st'][1]) + ' ' + str(data['mas_st'][2])
        file.write(wr_record)
        file.write(data['mas_destribution'][0])
        file.write(data['mas_destribution'][1])
        file.write(data['mas_destribution'][2])
        file.write(data['mas_values'][0])
        file.write(data['mas_values'][1])
        file.write(data['mas_values'][2])





def convert_to_JPEG(path):
    matrix, size = get_matrix_pixel(path)   #size = (height, width)
    matrix, mas_st = dc_level_shift(matrix, size)
    matrix = convert_image_to_YCbCr(matrix, size)
    matrix = wavelet(matrix, size)
    #шаг с квантованием
    mas_values, mas_destribution = mq_coder(matrix, size)


create_file({1:12, 2:24, 'fdg':25}, 2)
