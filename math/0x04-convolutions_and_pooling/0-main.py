#!/usr/bin/env python3

import numpy as np
import matplotlib 

matplotlib.use('Agg')
import matplotlib.pyplot as plt

convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.savefig('0-main1.png')
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()
    plt.savefig('0-main2.png')
