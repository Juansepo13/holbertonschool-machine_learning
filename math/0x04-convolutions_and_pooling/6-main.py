#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


pool = __import__('6-pool').pool


if __name__ == '__main__':

    dataset = np.load('../../supervised_learning/data/animals_1.npz')
    images = dataset['data']
    print(images.shape)
    images_pool = pool(images, (2, 2), (2, 2), mode='avg')
    print(images_pool.shape)

    plt.imshow(images[0])
    plt.show()
    plt.savefig('6-pool1.png')
    plt.imshow(images_pool[0] / 255)
    plt.show()
    plt.savefig('6-pool2.png')
