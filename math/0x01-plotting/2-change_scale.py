#!/usr/bin/env python3
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')
plt.yscale('log')
plt.xlim(0, 28650)
plt.show()
plt.savefig('2-change_scale.png')

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
