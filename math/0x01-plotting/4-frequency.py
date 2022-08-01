#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.locator_params('x', nbins=10)
plt.hist(student_grades, color='#228BC2',
         edgecolor='k', range=(0, 100))
plt.show()
