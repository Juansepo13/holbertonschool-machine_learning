#!/usr/bin/env python3
""" concatenates two matrices """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices """
    if (axis != 0 and axis != 1):
        return None
    matCon = []
    if (axis == 0):
        if (len(mat1[0]) != len(mat2[0])):
            return None
        for row1 in mat1:
            matCon.append(row1.copy())
        for row2 in mat2:
            matCon.append(row2.copy())
    elif (axis == 1):
        if (len(mat1) != len(mat2)):
            return None
        for i in range(len(mat1)):
            matCon.append([])
            matCon[i] = mat1[i] + mat2[i]

    return matCon
