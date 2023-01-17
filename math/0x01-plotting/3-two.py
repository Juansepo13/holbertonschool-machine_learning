#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.plot(x, y1, 'r--')
plt.plot(x, y2, 'g-')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.legend(['C-14', 'Ra-226'])
plt.show()
plt.savefig('3-two.png')


# #!/usr/bin/env python3
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