#!/usr/bin/env python3

# Type: project
# Teaches: algorithms, word-problems, loops


"""
The sieve of Eratosthenes is an algorithm for finding all prime numbers up to
a given limit. It is one of the most efficient algorithms for finding smaller
prime numbers, up to around 10 million. Every non-prime number between a given
prime 'p' and the next largest prime is composed of multiples of all primes
smaller or equal to 'p'. Simply put, the algorithm works because every
non-prime is a multiple of a smaller prime. Once you find all the non-primes
all that's left are primes. The algorithm can be stated as:

To find all prime numbers less than or equal to a given integer 'n':

1. Generate a list from the smallest prime (2) through 'n': (2, ..., n).
2. Let 'p' equal 2, the first number in the list.
3. Enumerate all multiples of 'p' less than or equal to 'n' and mark them:
   (2p, 3p, ...). Do not mark 'p'.
4. Find the next number in the list that is larger than 'p' and not marked. If
   there is no such number, stop. Otherwise let 'p' equal this number and
   repeat from step 3. This is the next prime.
5. When the algorithm terminates, all the unmarked numbers are all the primes
   less than or equal to 'n'.

Your job is to convert this into code.

Note: Wikipedia has a great visual overview of the algorithm but the
pseudocode there won't translate directly into Python.
"""

# The limit beyond which to stop searching.
n = 100

# The set contains the non-primes, i.e. the marked integers.
s = set()

# The list contains the primes, i.e. the solution.
l = list()

# The list from [2...n] (steps 1 & 2). Add 1 to include 'n' since
# 'range' uses an exclusive upper limit.
for p in range(2,n+1):
    # Find the next number (the for loop) that is not marked. Stop when
    # there are no unmarked numbers <= n. You have to check each number
    # from [2...n].  (step 4)
    if p in s:
        continue
    # Add the number to the list of primes (step 5). The original algorithm
    # has you go through the same list twice, first to mark numbers, then to
    # find the unmarked numbers. You can combine passes for efficiency.
    l.append(p)
    # Enumerate all multiples of 'p' (step 3).
    for i in range(2*p,n+1,p):
        s.add(i)


"""
The original algorithm can be made more efficient:

1. In step 3, rather than mark from '2p' onwards, you can mark from 'p*p'
   onwards since all the smaller multiples will have already been marked by
   previous primes.

2. In step 4 and as a consequence of the previous modification, rather than
   stop marking when 'p' exceeds 'n', stop when 'p*p' exceeds 'n'.

Incorporate these slight modifications into your original implementation.
"""

n = 100
s = set()
l = list()

for p in range(2,n+1):
    if p in s:
        continue
    l.append(p)
    # Stop marking when 'p*p' exceeds 'n' (modification 2)
    if p*p > n:
        continue
    # Mark from 'p*p' onwards (modification 1)
    for i in range(p*p,n+1,p):
        s.add(i)


"""
... and made even more efficient:

No even number beyond 2 can be prime. Therefore, don't include them in the
initial list. Don't mark their multiples and don't check them.

Incorporate this change into your code.
"""

n = 100
s = set()
l = [2]

for p in range(2+1,n+1,2):
    if p in s:
        continue
    l.append(p)
    if p*p > n:
        continue
    for i in range(p*p,n+1,p):
        s.add(i)
