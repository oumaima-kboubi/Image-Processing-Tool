import numpy as np

def linear_transformation(matrix):
    matrix = np.matrix(matrix)
    max = matrix.max()
    min = matrix.min()
    matrix_transformed = (255/(max-min))*(matrix-min)
    return matrix_transformed

def saturated_transformation(matrix, min, max):
    matrix = np.matrix(matrix)
    matrix_transformed = (255/(max-min))*(matrix-min)
    return matrix_transformed

