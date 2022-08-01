#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

""" configure the subplots, rows and columns """
plt0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1)
plt1 = plt.subplot2grid((3, 2), (0, 1), rowspan=1)
plt2 = plt.subplot2grid((3, 2), (1, 0), rowspan=1)
plt3 = plt.subplot2grid((3, 2), (1, 1), rowspan=1)
plt4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

""" global settings """
plt.rcParams.update({'axes.titlesize': 'x-small'})
plt.rcParams.update({'axes.labelsize': 'x-small'})

""" 0 """
plt.title('All in One')
plt0.plot(np.arange(0, 11), y0, 'r-')

""" 1 """
plt1.scatter(x1, y1, color='m')
plt1.set_title('Scatter Plot')
plt1.set_xlabel('Height (in)')
plt1.set_ylabel('Weight (lbs)')
plt1.set_title("Men's Height vs Weight")

""" 2 """
plt2.plot(x2, y2)
plt2.set_xlabel('Time (years)')
plt2.set_ylabel('Fraction Remaining')
plt2.set_title('Exponential Decay of C-14')
plt2.set_yscale('log')
plt2.set_xlim(0, 28650)

""" 3 """
plt3.plot(x3, y31, 'r--')
plt3.plot(x3, y32, 'g-')
plt3.set_xlabel('Time (years)')
plt3.set_ylabel('Fraction Remaining')
plt3.set_title('Exponential Decay of Radioactive Elements')
plt3.set_xlim(0, 20000)
plt3.set_ylim(0, 1)
plt3.legend(['C-14', 'Ra-226'])

""" 4 """
plt4.set_xlabel('Grades')
plt4.set_ylabel('Number of Students')
plt4.set_title('Project A')
plt4.set_xlim(0, 100)
plt4.set_ylim(0, 30)
plt4.locator_params('x', nbins=10)
plt4.hist(student_grades, bins=10, color='#228BC2',
          edgecolor='k', range=(0, 100))

""" pack and display plots """

plt.tight_layout()
plt.suptitle('All in One')
plt.show()
