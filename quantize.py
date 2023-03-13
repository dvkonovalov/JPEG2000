import numpy as np

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
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])


def quantize(matrix, n):
    """n: кэф квантования
       matrix: матрица полученная на 3 шаге"""
    new_matrix = np.array([[(0, 0, 0)]*len(matrix[0])]*len(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i, j, 0]/(q_y[i % 8][j % 8]*n))
            new_matrix[i, j, 0] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i][j][1]/(q_c[i % 8][j % 8]*n))
            new_matrix[i, j, 1] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i, j, 2]/(q_c[i % 8][j % 8]*n))
            new_matrix[i, j, 2] = k

    return new_matrix


def reverse_quantize(matrix, n):
    """n: кэф квантования
        matrix: матрица полученная на 3 шаге"""
    original_matrix = np.array([[(0, 0, 0)]*len(matrix[0])]*len(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i, j, 0] * (q_y[i % 8][j % 8] * n))
            original_matrix[i, j, 0] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i, j, 1] * (q_c[i % 8][j % 8] * n))
            original_matrix[i, j, 1] = k
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            k = round(matrix[i, j, 2] * (q_c[i % 8][j % 8] * n))
            original_matrix[i, j, 2] = k

    return original_matrix