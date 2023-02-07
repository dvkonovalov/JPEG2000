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
