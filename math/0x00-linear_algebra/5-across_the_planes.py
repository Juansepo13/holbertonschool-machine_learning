#!/usr/bin/env python3
""" add two matrices """


def add_matrices2D(mat1, mat2):
    """ add two matrices """

    if len(mat1[0]) != len(mat2[0]):
        return None

    sum = []
    for x in range(len(mat1)):
        sumSub = []
        for y in range(len(mat1[x])):
            sumSub.append(mat1[x][y] + mat2[x][y])
        sum.append(sumSub)
    return sum
