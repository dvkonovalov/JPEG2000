import numpy as np
def get_destribution():
    dest_y = {}
    dest_cb = {}
    dest_cr = {}
    pr = 0
    for i in range(-256, 513):
        dest_y[i] = [pr, pr + 1]
        dest_cb[i] = [pr, pr + 1]
        dest_cr[i] = [pr, pr + 1]
        pr += 1
    return [dest_y, dest_cb, dest_cr]


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
    distribution_massiv = get_destribution()
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3
    mas = [[], [], []]

    for rounds in range(3):
        for i in range(size[0]):
            distribution = distribution_massiv[rounds]
            delitel = distribution[512][1]
            le = 0
            h = 65535
            bits_to_follow = 0
            string1 = ''

            for j in range(size[1]):
                pixel = matrix[i, j]
                component = pixel[rounds]
                ln = le + (distribution[component][0] * (h - le + 1)) // delitel
                h = le + (distribution[component][1] * (h - le + 1)) // delitel - 1
                le = ln
                if le>h:
                    print(distribution[component])

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
                delitel = distribution[512][1]

            mas[rounds].append(string1)
    return mas


def mq_coder_revers(data_mas, size):
    """
    Арифметическое декодирование (обратный MQ-кодер)
    :param mas: Массив с закодированными последовательностями
    :param size: размеры получаемой матрицы в виде кортежа
    :return: матрица изображения после декодирования
    """
    matrix = np.array([[(0, 0, 0) for j in range(size[1])] for i in range(size[0])])
    distribution_massiv = get_destribution()
    first_qtr = 65536 // 4
    half = first_qtr * 2
    third_qtr = first_qtr * 3

    for rounds in range(3):
        mas = data_mas[rounds]

        for i in range(size[1]):

            distribution = distribution_massiv[rounds]
            delitel = distribution[512][1]
            string = mas[]
        l = 0
        h = 65535
        value = int(string[:16], 2)
        next_pos = 15
        height = 0
        width = 0
        while (next_pos < len(string)):
            freq = ((value - l + 1) * delitel - 1) // (h - l + 1)
            j = 0
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
                if (next_pos >= len(string)):
                    break
                value = value * 2 + int(string[next_pos])

            component_pixel = j
            pixel = matrix[height, width]
            pix = list(pixel)
            pix[rounds] = component_pixel
            pix = tuple(pix)
            matrix[height, width] = pix
            width += 1

            distribution = update_destribution(distribution, j)
            delitel = distribution[512][1]

            if (width == size[1]):
                height += 1
                width = 0
                if (height == size[0]):
                    break
    return matrix