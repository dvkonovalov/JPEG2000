from PIL import Image


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


def dc_level_shift(matrica, size, st):
    """
    Выполнение сдвига яркости изображения (DC level shift)
    :param matrica: матрица пикселей изображения
    :param size: кортеж размеров - (высота, ширина)
    :param st: кортеж значений степеней ST для как каждой компоненты - (R, G, B)
    :return: Матрица изображения после сдвига яркости
    """
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrica[i, j]
            for color in range(3):
                pixel[color] -= 2 ** (st[color] - 1)
            matrica[i, j] = pixel
    return matrica


def dc_level_shift_revers(matrica, size, st):
    """
    Выполняет возрат сдвига яркости в изображение
    :param matrica: матрица пикселей изображения со сдвигом яркости
    :param size: кортеж размеров - (высота, ширина)
    :param st: кортеж значений степеней ST для как каждой компоненты - (R, G, B)
    :return: Матрица изображения без сдвига яркости
    """
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = matrica[i, j]
            for color in range(3):
                pixel[color] += 2 ** (st[color] - 1)
            matrica[i, j] = pixel
    return matrica


# matrica, width, height = get_matrix_pixel('wood.jpg')
# for i in range(height):
#     for j in range(width):
#         pixel = matrica[j, i]
#         print(pixel)
