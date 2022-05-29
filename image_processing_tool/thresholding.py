import math
import numpy as np
from image_processing_tool.imageStats import histogram

def thresholding(f, L):
    # create a new image with zeros
    f_tr = np.zeros(f.shape).astype(int)
    # setting to 255 the pixels below the threshold
    f_tr[np.where(f > L)] = 255
    return f_tr 

def otsu(matrix):
    lx,ly = np.shape(matrix)
    matrix = np.matrix(matrix)
    histogramme = histogram(matrix)

    max = -math.inf
    minSeuil = -1
    
    for seuil in range(1,len(histogramme)-1):
        w1 = (np.sum(histogramme[:seuil]))
        w2 = (np.sum(histogramme[seuil:]))
        variance1 = np.var(histogramme[:seuil])
        variance2 = np.var(histogramme[seuil:])
        variance = w1 * variance1 + w2 * variance2
        
        if(variance>max):
            max = variance
            maxSeuil = seuil
            
    return(thresholding(matrix, maxSeuil), maxSeuil)

def pad(matrix):
    matrix = np.matrix(matrix)
    matrix_padded = np.pad(matrix, pad_width=1, mode='constant', constant_values=0)
    return matrix_padded

def dilatation(matrix):
    padded_matrix = pad(matrix)
    lx,ly = np.shape(padded_matrix)
    new_padded_matrix = np.zeros((lx,ly)).astype(int)
    for x in range(1,lx-1):
        for y in range(1,ly-1):
            arr = np.array([padded_matrix[x-1,y-1],
                   padded_matrix[x,y-1],
                   padded_matrix[x+1,y-1],
                   padded_matrix[x-1,y],
                   padded_matrix[x,y],
                   padded_matrix[x+1,y],
                   padded_matrix[x-1,y+1],
                   padded_matrix[x,y+1],
                   padded_matrix[x+1,y+1]])
            min = arr.min()
            new_padded_matrix[x,y] = min          
    new_unpadded_matrix = np.delete(new_padded_matrix, lx-1, 0)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, ly-1, 1)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, 0, 0)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, 0, 1)
    return new_unpadded_matrix

def errosion(matrix):
    padded_matrix = pad(matrix)
    lx,ly = np.shape(padded_matrix)
    new_padded_matrix = np.zeros((lx,ly)).astype(int)
    for x in range(1,lx-1):
        for y in range(1,ly-1):
            arr = np.array([padded_matrix[x-1,y-1],
                   padded_matrix[x,y-1],
                   padded_matrix[x+1,y-1],
                   padded_matrix[x-1,y],
                   padded_matrix[x,y],
                   padded_matrix[x+1,y],
                   padded_matrix[x-1,y+1],
                   padded_matrix[x,y+1],
                   padded_matrix[x+1,y+1]])
            min = arr.max()
            new_padded_matrix[x,y] = min          
    new_unpadded_matrix = np.delete(new_padded_matrix, lx-1, 0)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, ly-1, 1)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, 0, 0)
    new_unpadded_matrix = np.delete(new_unpadded_matrix, 0, 1)
    return new_unpadded_matrix

def opening(matrix):
    return errosion(dilatation(matrix))

def closing(matrix):
    return dilatation(errosion(matrix))

