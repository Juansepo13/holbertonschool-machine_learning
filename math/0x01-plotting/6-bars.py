#!/usr/bin/env python3
import numpy as np
import matplotlib 

matplotlib.use('Agg')

import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))
people = ['Farrah', 'Fred', 'Felicia']
prod = ['apples', 'bannanas', 'orange', 'peaches']
color = ['r', 'k', '#ff8000', '#ffe5b4']
width = 0.5

fig, ax = plt.subplots()

for f in fruit:
    print(f)

p1 = ax.bar(people, fruit[0], width, label='apples', color='r')
p2 = ax.bar(people, fruit[1], width, bottom=fruit[0],
            label='bannanas', color='yellow')
p3 = ax.bar(people, fruit[2], width, bottom=fruit[0]+fruit[1],
            label='orange', color='#ff8000')
p4 = ax.bar(people, fruit[3], width, bottom=fruit[0]+fruit[1]+fruit[2],
            label='peaches', color='#ffe5b4')
ax.set_ylabel('Quantity of Fruit')
ax.set_ylim(0, 80)
ax.legend()
plt.show()
plt.savefig('6-bars.png')
