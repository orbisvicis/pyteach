#!/usr/bin/env python3

# Type: competition
# Teaches: types, lists, nested-lists, list-indexing, list-slicing

import readline
import random
import ast
import re
import collections


def print_score(ri, rt):
    print(f"\nRound {ri} of {rt} finished. The score is:")
    w = max(len(p) for p in players) + 4
    for i,p in enumerate(players):
        print(f"\t{p:{w}}: {scores[i]:2}")

def input_something(p):
    while True:
        s = input(p)
        if s and not s.isspace():
            return s

def try_again(p="Try again"):
    while True:
        r = input(p + " [yn]? ")
        if r.lower() in ["y","yes"]:
            return True
        if r.lower() in ["n","no"]:
            return False

def element_at_find(o):
    s = random.sample([True, False], k=1, counts=[5,1])[0]
    if isinstance(o, list) and o and s:
        n = random.randrange(-len(o), len(o))
        r = element_at_find(o[n])
        r[0].appendleft(n)
        return r
    return (collections.deque(), o)

def element_at(pi, o, ts):
    print()
    ix, e = element_at_find(o)
    s = "l" + "".join(f"[{i}]" for i in ix)
    g = input(f"What is the element at {s}? ")
    try:
        g = ast.literal_eval(g)
    except Exception:
        print("That's not valid Python code")
        v = False
    else:
        v = (g == e)
    if v:
        scores[pi] += 1
        print(f"Correct. One point to {players[pi]}!")
    else:
        scores[1-pi] += 1
        print(f"Incorrect. One point to {players[1-pi]}!")
        print(f"The correct answer is: '{e}'")
    return True

def flatten(i, o):
    if isinstance(o, list) and o:
        f = []
        for j,v in enumerate(o):
            f.extend(flatten(i+[(j,j-len(o))], v))
        return f
    return [(i, o)]

def index_of(pi, o, ts):
    if not o:
        return False
    print()
    f = flatten([], o)
    i = random.randrange(len(f))
    ix,e = f[i]
    n = sum(1 for v in f[:i] if v[1] == e) + 1
    sx = ["st","nd","rd","th"]
    s = sx[-1] if n > len(sx) else sx[n-1]
    while True:
        g = input(f"What is a possible indexing for the {n}{s} '{e}' "
                   "(list of integers)? ")
        r = r"[+-]?\d+"
        g = [int(v) for v in re.findall(r, g)]
        if not try_again(f"Your answer: {g}. Input a different answer"):
            break
    if len(ix) != len(g) or not all((ig in ic) for ic,ig in zip(ix, g)):
        c = [random.choice(i) for i in ix]
        scores[1-pi] += 1
        print(f"Incorrect. One point to {players[1-pi]}!")
        print(f"A correct answer would have been: {c}")
    else:
        scores[pi] += 1
        print(f"Correct. One point to {players[pi]}!")
    return True

def format_slice(s):
    m = "["
    if s.start is not None:
        m += str(s.start)
    m += ":"
    if s.stop is not None:
        m += str(s.stop)
    if s.step is not None and s.step != 1:
        m += ":" + str(s.step)
    m += "]"
    return m

def slice_of(pi, o, ts):
    print()
    slices = []
    while True:
        t = [True, False]
        b = random.betavariate(4,2)
        w = round(len(o) * b)
        e = round(len(o) * 0.5)
        m = random.randrange(len(o) - w + 1)
        s = [0 + m, w + m, 1]
        if random.choice(t):
            s[0] -= len(o)
        if random.choice(t) and s[1] != len(o):
            s[1] -= len(o)
        if s[0] == -len(o) and random.choice(t):
            s[0] -= random.randrange(0, e + 1)
        elif s[0] == 0 and random.choice(t):
            s[0] = None
        if s[1] == len(o) and random.choice([True,False,False]):
            s[1] = None
        elif s[1] == len(o) and random.choice(t):
            s[1] += random.randrange(0, e + 1)
        if random.choices(t, weights=[1,2], k=1)[0]:
            s = [ s[1] if s[1] is None else s[1] - 1
                , s[0] if s[0] is None else s[0] - 1
                , -s[2]
                ]
        if None in [s[0],s[1]] and random.choices(t, weights=[1,2], k=1)[0]:
            s[2] *= -1
        if random.choices(t, weights=[1,9], k=1)[0]:
            s[2] *= -1
        c = [1,1] if (w > 4 and b > 0.6) else [6,1]
        s[2] *= random.sample([1,2], 1, counts=c)[0]
        s = slice(*s)
        slices.append(s)
        o = o[s]
        # 1/2 chance of stopping if remaining
        # 3/4 chance of stopping if empty
        m = [1,1] if o else [3,1]
        if random.sample(t, 1, counts=m)[0]:
            break
    ss = "l" + "".join(format_slice(s) for s in slices)
    g = input(f"What is the value of {ss}? ")
    try:
        g = ast.literal_eval(g)
    except Exception:
        print("That's not valid Python code")
        v = False
    else:
        v = (g == o)
    if v:
        scores[pi] += 1
        print(f"Correct. One point to {players[pi]}!")
    else:
        scores[1-pi] += 1
        print(f"Incorrect. One point to {players[1-pi]}!")
        print(f"The correct answer is: {o}")
    return True

def consume(pi, v, o, ts, count=4):
    print("")
    if not v:
        print(f"{players[pi]}, turn complete - "
               "your opponent failed to generate any challenges.")
        return
    print(f"{players[pi]}, you must answer {count} questions given: ")
    print(f"{'':2}l = {o}")
    ss = [element_at, index_of, slice_of]
    while count:
        if random.choice(ss)(pi, o, ts):
            count -= 1

def validate(o, ts):
    w = len(str(o)) + 1
    q = collections.deque()
    q.append((o, 0))
    p = None
    while q:
        o,i = q.popleft()
        if i != p:
            p = i
            print(f"{'':2}Checking '{ts[i].__name__}':")
        print(f"{'':4}{o!s:<{w}}", end="")
        if not isinstance(o, ts[i]):
            print(" <-- bad")
            return False
        print("... pass", end="")
        if isinstance(o, list) and not o and i < len(ts) - 1:
            t = ", ".join(t.__name__ for t in ts[i+1:])
            print(f" (omitted [{t}] which is technically correct)")
        else:
            print("")
        if not isinstance(o, list):
            continue
        for e in o:
            q.append((e, i+1))
    return True

def types_name(ts):
    return "[".join(t.__name__ for t in ts) + "]" * (len(ts) - 1)

def generate(pi, attempts=4):
    ss = [list, int, float, str]
    ts = [list]
    while True:
        t = random.choices(ss, weights=[3,2,2,2], k=1)[0]
        ts.append(t)
        if t is not list:
            break
    n = types_name(ts)
    while attempts:
        o = input(f"\n{players[pi]}, give me a {n}: ").strip()
        try:
            o = ast.literal_eval(o)
        except Exception:
            print(f"{'':2}That's not valid Python code")
            v = False
        else:
            v = validate(o, ts)
        print("")
        if v:
            scores[pi] += 1
            print(f"Correct. One point to {players[pi]}!")
            return (True, o, ts)
        scores[1-pi] += 1
        attempts -= 1
        print(f"Incorrect. One point to {players[1-pi]}!")
        if attempts and not try_again(f"Try again ({attempts} remaining)"):
            break
    if not attempts:
        print("All attempts exhausted, turn complete")
    return (False, None, ts)


players =\
    [ input_something("Player 1, what is your name? ")
    , input_something("Player 2, what is your name? ")
    ]

scores = [0, 0]

while True:
    try:
        rounds = int(input("How many rounds (must be even)? "))
    except ValueError:
        continue
    if rounds <= 0:
        continue
    if rounds % 2:
        continue
    break

pi = random.randrange(len(players))

print(f"\nOK, let's play {rounds} rounds. {players[pi]} will go first")

for r in range(1, rounds+1):
    consume(1-pi, *generate(pi))
    print_score(r, rounds)
    pi = 1-pi

print("\nThe games have concluded. Thank you for your participation")
