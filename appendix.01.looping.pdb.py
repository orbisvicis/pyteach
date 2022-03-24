#!/usr/bin/env python3

# Type: visualization
# Teaches: loops, assignments

import random
import inspect
import bdb

try:
    import ipdb as pdbm
except ModuleNotFoundError:
    import pdb as pdbm

import pygments
import pygments.lexers
import pygments.formatters


registry = {}

def register(name=None):
    def inner(f):
        nonlocal name
        if name is None:
            name = f.__name__.replace("_", " ")
        if name in registry or f in registry.values():
            raise KeyError("Duplicate Entries")
        registry[name] = f
        return f
    return inner

def filter_trace(src):
    src = src.splitlines()
    n = 0
    for i,l in enumerate(src):
        if "set_trace" in l:
            n = i + 1
            break
    while n < len(src) and not src[n]:
        n += 1
    return "\n".join(src[n:])

@register()
def summation():
    """
    Find the sum of a non-empty list of integers using a for loop.
    """

    pdbm.set_trace()

    data = random.choices(range(100), k=15)

    initial = 0

    for entry in data:
        initial = initial + entry

    print(initial)

    return initial

@register()
def summation_with_offset():
    """
    Startin from 10, find the sum of a non-empty list of integers using a for
    loop.
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=10)

    initial = 10

    for entry in data:
        initial = initial + entry

    print(initial)

    return initial

@register("convert & append")
def conv_append():
    """
    Concatenate a non-empty list of integers using a for loop, separated by
    commas.
    """

    pdbm.set_trace()

    data = random.choices(range(100), k=15)

    initial = ""

    for entry in data:
        initial = initial + str(entry)

    print(initial)

    return initial

@register()
def partial_sums():
    """
    Using a for loop, create a list of intermediate sums, each entry being the
    sum from zero for that entry and all previous entries. For example:

    [10, 3, 0, 2, 9]  ->  [10, 13, 13, 15, 24]
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=10)

    initial = []

    for entry in data:
        if initial:
            initial.append(initial[-1] + entry)
        else:
            initial.append(entry)
    print(initial)

    return initial

@register()
def partial_max():
    """
    Using a for loop, create a list of intermediate maximums, each entry being
    the maximum entry to-date. For example:

    [1, 3, 0, 2, 9, 5]  ->  [1, 3, 3, 3, 9, 9]
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=10)

    initial = []

    for entry in data:
        if initial:
            initial.append(max(initial[-1], entry))
        else:
            initial.append(entry)
    print(initial)

    return initial

    pass

@register()
def square():
    """
    Find the square of each entry in a non-empty list of integers using a for
    loop. For example:

    [1, 2, 3, 4, 5, 6]  ->  [1, 4, 9, 16, 25, 36]
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=10)

    initial = []

    for entry in data:
        initial.append(entry**2)

    print(initial)

    return initial

@register()
def count():
    """
    Find the length of the list using a for loop.
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=10)

    initial = 0

    for entry in data:
        initial += 1

    print(initial)

    return initial

@register()
def reverse():
    """
    Reverse a sequence using a for loop. For example:

    "this is awesome"  ->  "emosewa si siht"

    Notes:
        1. See "03.list.append.time.py" for performance considerations.
        2. This could easily be achieved, and more performant, with [::-1].
    """

    pdbm.set_trace()

    data = random.choices(range(48,91), k=10)
    data = "".join(chr(i) for i in data)

    initial = ""

    for entry in data:
        initial = entry + initial

    print(initial)

    return initial

@register()
def reverse_while_index():
    """
    Reverse a sequence using a while loop with an index variable.
    """

    pdbm.set_trace()

    data = random.choices(range(48,91), k=10)
    data = "".join(chr(i) for i in data)

    reverse = ""
    index = len(data)-1

    while index >= 0:
        reverse += data[index]
        index -= 1

    print(reverse)

    return reverse

@register()
def until_limit():
    """
    Print each list item until the sum of items exceeds 100.
    """

    pdbm.set_trace()

    data = random.choices(range(10), k=1000)

    initial = 0

    for entry in data:
        initial += entry
        if initial > 100:
            break
        print(f"The current value is: {entry}")

    print(f"The total sum was: {initial}")

    return initial

@register()
def filter_small_lists():
    """
    Using a for loop, remove all sub-lists which have five or fewer elements.
    For example:

    [[1, 2, 3], [], [4, 5, 7, 9, 21, 6, 6], [22, 11, 3, 3, 5, 9, 0], [1]]
    ...
    [[4, 5, 7, 9, 21, 6, 6], [22, 11, 3, 3, 5, 9, 0]]
    """

    data = []
    for _ in range(10):
        l = random.randrange(0,10)
        data.append(list(range(l)))

    pdbm.set_trace()

    data = data

    initial = []

    for entry in data:
        if len(entry) <= 5:
            continue
        initial.append(entry)

    print(initial)

    return initial

@register("delete list entries, INCORRECT #1. Why?")
def delete_list_entries_bad_1():
    """
    NOTE: THIS IS AN EXAMPLE OF INCORRECT CODE

    Here we attempt to remove all entries from a list by value. Instead, we
    remove every other entry. Can you figure out why this is happening?

    [0, 1, 2, 4, 5, 6, 7, 8, 9]
    ...
    []
    """

    pdbm.set_trace()

    data = list(range(10))

    for entry in data:
        data.remove(entry)

    print(data)

    return data

@register("delete list entries, INCORRECT #1. Solution #1 - slow")
def delete_list_entries_bad_1_sol_slow():
    """
    NOTE: THIS IS AN EXAMPLE OF INEFFICIENT CODE

    Once again we attempt to remove all entries from a list by value. We know
    what went wrong last time, and how to fix it. While this solution works, it
    is very inefficient. Can you figure out why?
    """

    pdbm.set_trace()

    data = list(range(10))

    for entry in data[:]:
        data.remove(entry)

    print(data)

    return data

@register("delete list entries, INCORRECT #2. Why?")
def delete_list_entries_bad_2():
    """
    NOTE: THIS IS AN EXAMPLE OF INCORRECT CODE

    We know from previous examples that removing list entries by value is
    inefficient. Let's try removing all entries from a list by index. As before
    we make a copy of the list to avoid the same incorrect solution as last
    time. But this is an error! Can you figure out why?
    """

    pdbm.set_trace()

    data = list(range(10))

    for index,_ in enumerate(data[:]):
        del data[index]

    print(data)

    return data

@register("delete list entries, INCORRECT #2. Solution #1 - fast")
def delete_list_entries_bad_2_sol_fast():
    """
    Now that we understand the internal structure of Python lists, we know how
    to properly remove entries by index. We're smart enough not to saw off the
    branch on which we sit.

    NOTE: THIS IS AN EXAMPLE OF UNUSUAL CODE

    The most Pythonic approach is to create a new list, then replace the
    contents of the old list with the new list. Usually this is done with a
    list comprehension, and is included here for the sake of completeness. By
    assigning to a slice we guarantee the original list is modified, rather
    than replaced with a copy. While not the most efficient, it is fast enough
    and most importantly, easy to read.

    data[:] = [i for i in data if some_condition(i)]

    Of course, since we are clearing rather than filtering the list, we can
    just use one of the built-in list methods:

    data.clear()
    """

    pdbm.set_trace()

    data = list(range(10))

    for index in range(len(data)-1, -1, -1):
        del data[index]

    print(data)

    return data

@register("delete list entries, INCORRECT #2. Solution #2 - slow. Why?")
def delete_list_entries_bad_2_sol_slow():
    """
    We've decided to simplify the code from the previous example. Yet now it
    runs much more slowly. Why is this?
    """

    pdbm.set_trace()

    data = list(range(10))

    for _ in range(len(data)):
        del data[0]

    print(data)

    return data

@register()
def minimum():
    """
    Use a for loop to find the smallest entry in a list.
    """

    pdbm.set_trace()

    data = random.choices(range(100), k=15)

    initial = None

    for entry in data:
        if initial is None or entry < initial:
            initial = entry

    print(initial)

    return initial

@register()
def prefix_strings():
    """
    Including the empty string, use a for-loop to find all the possible prefix substrings. For example:

    "ate": ["", "a", "at", "ate"]
    """

    pdbm.set_trace()

    data = random.choices(range(48,91), k=10)
    data = "".join(chr(i) for i in data)

    initial = [""]

    for entry in data:
        initial.append(initial[-1] + entry)

    print(initial)

    return initial

@register()
def while_random():
    """
    While a random number is less than 80, print that number.
    """

    pdbm.set_trace()

    while True:
        data = random.randint(0,100)

        if data >= 80:
            break

        print(data)

    print(f"The final (unseen) random number was: {data}")

    return data

@register()
def for_file_lines():
    """
    Open this file in text read mode. Use a for loop to print every line prefixed by the line number.
    """

    pdbm.set_trace()

    index = 0

    with open(__file__, "r") as f:
        for line in f:
            print(str(index) + ": " + line, end="")
            index += 1

@register()
def while_file():
    """
    Open this file in binary read mode. Read the file in chunks of 1KB until
    the end of the file is reached.

    Note: "If 0 bytes are returned [by read(size)], and size was not 0, this
    indicates end of file". See:
    * https://docs.python.org/3/library/io.html#io.RawIOBase.read
    """

    pdbm.set_trace()

    with open(__file__, "rb") as f:
        while True:
            d = f.read(1024)
            if not d:
                print("End-of-File reached")
                break
            d = d[:100] + b"..."
            print(f"Current chunk:\n{d}\n")


def select(registry):
    w = len(registry)
    l = list(registry.values())
    for i,k in enumerate(registry.keys()):
        print(f"{i:{w}}: {k}")
    print()
    while True:
        s = input("Selection (name or number)? ")
        try:
            f = l[int(s)]
        except (ValueError,IndexError):
            pass
        else:
            break
        try:
            f = registry[s]
        except KeyError:
            pass
        else:
            break
        print("Please enter a valid selection")
    if f.__doc__:
        doc = "\n".join(l.strip() for l in f.__doc__.strip().splitlines())
        print(f"\n{doc}\n\n")
    else:
        print()
    source = filter_trace(inspect.getsource(f))
    source = pygments.highlight\
        ( source
        , pygments.lexers.PythonLexer()
        , pygments.formatters.TerminalFormatter()
        )
    print(f"{source}\n\n")
    try:
        return f()
    except bdb.BdbQuit:
        pass


if __name__ == "__main__":
    print\
        ( "Step-by-step interactive debugging (visualization) "
          "of simple examples:\n"
        )
    select(registry)
