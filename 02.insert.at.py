#!/usr/bin/env python3

# Type: exercise
# Teaches: list-indexing, list-slicing, list-comparison


l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""
Use only slicing.

1. Insert 100 between 1 and 2.
"""

l[2:2] = [100]

"""
2. Insert 99 at the very beginning.
"""

l[0:0] = [99]

"""
3. Insert 98 at the very end.
"""

l[len(l):len(l)] = [98]

"""
4. Insert 97 before 98.
"""

l[-1:-1] = [97]

"""
5. Replace [2,3,4] with [20,30,40] in a single operation.
"""

l[4:7] = [20,30,40]

"""
6. Assign a copy of the list to the variable 'b'.
"""

b = l[:]

"""
7. Replace 100 with -100.
"""

l[3:4] = [-100]

"""
8. Did 'b' change? What is 'l == b'?
"""

False

"""
9. Starting with 0, replace every third entry with the string "cat", in a
single operation.
"""

l[1::3] = ["cat"]*5

"""
10. Same as 9 but with fewer cats. Keep the first cat but replace the remaining with: squirrel, dog, chipmunk, bear.
"""

l[4::3] = ["squirrel", "dog", "chipmunk", "bear"]

"""
11. Is your list equal to this?
"""

answer =\
    [ 99, 'cat', 1, -100, 'squirrel', 30, 40
    , 'dog', 6, 7, 'chipmunk', 9, 97, 'bear'
    ]

l == answer
