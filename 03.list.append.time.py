#!/usr/bin/env python3

# Type: exercise
# Teaches: data-structures, lists


"""
Inspect the two following code segments being timed. They are indentical in all respects but that the first appends to the end of the list while the second prepends to the beginning of the list.

Each segment is run 1000 times. The append approach takes roughly 9 seconds. The prepend approach takes roughly 2150 seconds - roughly 246 times longer. Why do you think this is?
"""

append="""
l = []
for i in range(10**5):
    l.append(i)
"""
>>> timeit.timeit(append, number=10**3)
8.81333849998191

prepend="""
l = []
for i in range(10**5):
    l.insert(0, i)
"""
>>> timeit.timeit(prepend, number=10**3)
2167.546522699995


"""
Imagine a list to be like a row of empty slots for holding objects such as billiard balls. To add to the end of the list, just drop the ball into the first empty slot. If there are no empty slots, create a bunch of empty slots (say 100) in computer memory.

Now imagine you have a row containing 10 billiard balls but you need to drop a ball into the first slot. You need to make room. You could bump the first ball to the back of the row, but that will mix the ordering and that's very bad. So you move the 10th ball one slot to the right, the 9th ball to the now-empty slot to the right, ... until the first slot is free. Say you need to add ten balls to the front:

01. Shift 10 balls to the right: 10 operations
02. Shift 11 balls to the right: 11 operations
...
10. Shift 19 balls to the right: 19 operations

So you total operations is 19+18 ... + 11+10 = 145 operations. Compare that to 10 operations to simply append.
"""
