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
            return 0.82389726084544
        elif a == 1:
            return 0.2937295293203774
        elif a == 2:
            return -0.0274753175870262
        elif a == 3:
            return -0.054415842243081
        else:
            return 0
    else:
        if a == 0:
            return 0.2878479639221569
        elif a == 1:
            return -0.1314931795880806
        elif a == 2:
            return -0.0400426318022275
        elif a == 3:
            return 0.0902040104318745
        elif a == 4:
            return 0.0145213947622878
        else:
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
                y += sum([image[i, columns_count - k][0] * hl(distance - k, imparity) for k in range(1, count)])
                cb = sum([image[i, j + k][1] * hl(k, imparity) for k in range(-4, columns_count - j)])
                cb += sum([image[i, columns_count - k][1] * hl(distance - k, imparity) for k in range(1, count)])
                cr = sum([image[i, j + k][2] * hl(k, imparity) for k in range(-4, columns_count - j)])
                cr += sum([image[i, columns_count - k][2] * hl(distance - k, imparity) for k in range(1, count)])
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
                y += sum([intermediate_2[rows_count - k, j][0] * hl(distance - k, imparity) for k in range(1, count)])
                cb = sum([intermediate_2[i + k, j][1] * hl(k, imparity) for k in range(-4, rows_count - i)])
                cb += sum([intermediate_2[rows_count - k, j][1] * hl(distance - k, imparity) for k in range(1, count)])
                cr = sum([intermediate_2[i + k, j][2] * hl(k, imparity) for k in range(-4, rows_count - i)])
                cr += sum([intermediate_2[rows_count - k, j][2] * hl(distance - k, imparity) for k in range(1, count)])
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
                y += sum([intermediate_1[rows_count - k, j][0] * gl(distance - k, imparity) for k in range(1, count)])
                cb = sum([intermediate_1[i + k, j][1] * gl(k, imparity) for k in range(-4, rows_count - i)])
                cb += sum([intermediate_1[rows_count - k, j][1] * gl(distance - k, imparity) for k in range(1, count)])
                cr = sum([intermediate_1[i + k, j][2] * gl(k, imparity) for k in range(-4, rows_count - i)])
                cr += sum([intermediate_1[rows_count - k, j][2] * gl(distance - k, imparity) for k in range(1, count)])
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
                y += sum([intermediate_3[i, columns_count - k][0] * gl(distance - k, imparity) for k in range(1, count)])
                cb = sum([intermediate_3[i, j + k][1] * gl(k, imparity) for k in range(-4, columns_count - j)])
                cb += sum([intermediate_3[i, columns_count - k][1] * gl(distance - k, imparity) for k in range(1, count)])
                cr = sum([intermediate_3[i, j + k][2] * gl(k, imparity) for k in range(-4, columns_count - j)])
                cr += sum([intermediate_3[i, columns_count - k][2] * gl(distance - k, imparity) for k in range(1, count)])
            result[i, j] = (y, cb, cr)
    return result


def transform(image, size, count, without_loss=True):
    rows_count = size[0]
    columns_count = size[1]
    if without_loss == True:
        image = wavelet_without_loss(image, size)
    else:
        image = wavelet_with_loss(image, size)
    print(rows_count, columns_count)
    k = 1
    while k < count:
        rows_count = ceil(rows_count / 2)
        columns_count = ceil(columns_count / 2)
        print(rows_count, columns_count)
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