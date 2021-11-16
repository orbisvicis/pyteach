#!/usr/bin/env python3

# Type: exercise
# Teaches: loops, for-loops


"""
l = [1,2,3,4]

Given the product of all possible three-way combinations from 'l', how many combinations are greater than 32?
"""

l = [1,2,3,4]
a = []
for i in l:
    for j in l:
        for k in l:
            a.append(i*j*k)

sum(1 for i in a if i > 32)


"""
You have a coin that has 1/2 chance of heads, a coin that has 1/3 chance of heads, and a coin that has 3/4 chance of heads. What is the probability that every coin is heads? Simulate it with 10**3 iterations.

Note: While technically you need an independent random number stream per coin, it is not necessary and for simplification can be removed.

import random
rgen1 = random.Random()
rgen2 = random.Random()
rgen3 = random.Random()
"""

import random
rgen1 = random.Random()
rgen2 = random.Random()
rgen3 = random.Random()

j = 0
n = 10**3
for i in range(n):
    o1 = rgen1.choice([True,False])
    o2 = rgen2.choice([True,False,False])
    o3 = rgen3.choice([True,True,True,False])
    if o1 and o2 and o3:
        j += 1
answer = round(j/n)

# Approximately 12.5%
