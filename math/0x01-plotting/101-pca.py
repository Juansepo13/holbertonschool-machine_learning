#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Figure 3D
fig = plt.figure()
axes = Axes3D(fig)

shift = np.stack(pca_data, axis=-1)
x, y, z = shift
axes.scatter(x, y, z, c=labels, cmap=plt.cm.plasma)

axes.set_xlabel("U1")
axes.set_ylabel("U2")
axes.set_zlabel("U3")
plt.title("PCA of Iris Dataset")
plt.show()
plt.savefig('101-pca.png')

# import numpy as np
# import matplotlib

# matplotlib.use('Agg')

# import matplotlib.pyplot as plt

# mean = [69, 0]
# cov = [[15, 8], [8, 15]]
# np.random.seed(5)
# x, y = np.random.multivariate_normal(mean, cov, 2000).T
# y += 180

# plt.scatter(x, y, color='m')
# plt.xlabel('Height (in)')
# plt.ylabel('Weight (lbs)')
# plt.title("Men's Height vs Weight")
# plt.show()
# plt.savefig('1-scatter.png')
