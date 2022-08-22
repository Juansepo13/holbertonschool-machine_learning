#!/usr/bin/env python3
""" Function def matrix_shape(matrix) that calculates the shape of a matrix"""


def matrix_shape(matrix):
    """ Calculate the shape of a matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
