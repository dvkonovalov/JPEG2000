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
            distribution_cb[pixel[0]][1] += 1
            distribution_cr[pixel[0]][1] += 1
    #Сортируем полученные массивы распределений
    distribution_y.sort(key=lambda x: x[1], reverse=True)
    distribution_cb.sort(key=lambda x: x[1], reverse=True)
    distribution_cb.sort(key=lambda x: x[1], reverse=True)

    distribution_y[0][2] = distribution_y[0][1]
    distribution_cb[0][2] = distribution_cb[0][1]
    distribution_cr[0][2] = distribution_cr[0][1]

    for i in range(1, 256):
        distribution_y[i][2] = distribution_y[i-1][2]+distribution_y[i][1]
        distribution_cb[i][2] = distribution_cb[i-1][2]+distribution_cb[i][1]
        distribution_cr[i][2] = distribution_cr[i-1][2]+distribution_cr[i][1]

    #переделываем массивы в словари для удобства использования в дальнейшем
    dist_y = {}
    dist_cb = {}
    dist_cr = {}
    for element in distribution_y:
        dist_y[element[0]] = element[2]
    for element in distribution_cb:
        dist_cb[element[0]] = element[2]
    for element in distribution_cr:
        dist_cr[element[0]] = element[2]
    return (dist_y, dist_cb, dist_cr)


def mq_coder(matrix, size):
    """
    Арифметическое кодирование (MQ-кодер)
    :param matrix: матрица пикселей после квантования
    :param size: кортеж с размерами изображения - (высота, ширина)
    :return: массив со значениями данных после арифметического кодирования,
    кортеж с распределениями из функции get_destribution
    """
    h = 65535
    distribution_y, distribution_cb, distribution_cr  = get_destribution(matrix, size)
    delitel = size[0]*size[1]
    first_qtr = (h+1)//4
    half = first_qtr*2
    third_qtr = first_qtr*3

    mas = []

    for round in range(3):
        string = ''
        l = 0
        h = 65535
        bits_to_follow = 0
        for i in range(size[0]):
            for j in range(size[1]):
                pixel = matrix[i, j]
                component = pixel[round]
                l = l + distribution_y[component-1]*(h-l+1)/delitel
                h = l + distribution_y[component]*(h-l+1)//delitel-1
                while (True):
                    if (h<half):
                        string += '0' + '1'*bits_to_follow
                        bits_to_follow = 0
                    elif (l>=half):
                        string += '1' + '0'*bits_to_follow
                        bits_to_follow = 0
                        l -= half
                        h -= half
                    elif ((l>=first_qtr) and (h< third_qtr)):
                        bits_to_follow += 1
                        l -= first_qtr
                        h -= first_qtr
                    else: break
                    l += l
                    h += h+1
        mas.append(string)
    return mas, (distribution_y, distribution_cb, distribution_cr)


def mq_coder_revers(mas, size):
    """
    Арифметическое декодирование (обратный MQ-кодер)
    :param mas: Массив с закодированными последовательностями
    :param size: размеры получаемой матрицы в виде кортежа
    :return: матрица изображения после декодирования
    """
    matrix = Image.new('RGB', size, 'white')




# matrica, width, height = get_matrix_pixel('wood.jpg')
# for i in range(height):
#     for j in range(width):
#         pixel = matrica[i, j]
#         print(pixel)
