#!/usr/bin/env python3
""" adds two arrays """


def add_arrays(arr1, arr2):
    """ add two arrays element-wise """
    add = []
    if len(arr1) != len(arr2):
        return None
    for x in range(len(arr1)):
        add.append(arr1[x] + arr2[x])
    return add
