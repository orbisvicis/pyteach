#!/usr/bin/env python3

# Type: exercise
# Teaches: data-model, data-structures, nested lists


a = [1,2,3]
b = [a, a, a]
a.append(4)
b[-1].append(44)

"""
What do you think 'b' will look like, and why?
"""

[ [1, 2, 3, 4, 44]
, [1, 2, 3, 4, 44]
, [1, 2, 3, 4, 44]
]


a = [1,2,3]
b = [a, a, a]
a = [4, 5, 6]
b[-1].append(44)

"""
What do you think 'b' will look like, and why?
"""

[ [1, 2, 3, 44]
, [1, 2, 3, 44]
, [1, 2, 3, 44]
]


"""
Each name in python is bound to an object, so in essence the second list contains three references to the same object. Since the object is the same, modification of any entry is reflected in all the others.

A variable name is just a binding to an object. When you assign to that name, you don't modify the object, you simply bind that variable to a new object, a new list. Meanwhile, the objects bound by the list remain unchanged.
"""
