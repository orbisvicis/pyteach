#!/usr/bin/env python3

# Type: project
# Teaches: functions, command-line arguments, string formatting

import argparse


def positive(s):
    s = int(s)
    if s <= 0:
        raise ValueError
    return s

def phrase(s):
    s = " ".join(s.split())
    if not s:
        raise ValueError
    if not all(c.isalpha() or c.isspace() for c in s):
        raise ValueError
    return s


parser = argparse.ArgumentParser(description="Play Hangman")
parser.add_argument\
    ( "phrase"
    , type=phrase
    , help="phrase to guess, must not be empty"
    )
parser.add_argument\
    ("-l", "--limit"
    , type=positive
    , default=10
    , help="guess limit, must be a natural number"
    )

args = parser.parse_args()

indices = {}

for i,c in enumerate(args.phrase):
    c = c.lower()
    if c.isspace():
        continue
    if c not in indices:
        indices[c] = []
    indices[c].append(i)

mask = ["_" if c.isalpha() else c for c in args.phrase]

seen = set()

m = "Let's play hangman! {} guess{} left"
m = m.format(args.limit, "" if args.limit == 1 else "es")
print(m, end="\n\n")

while args.limit and indices:
    print(f"Hangman says: {''.join(mask)}")
    g = input("Guess: ")
    if not g:
        print("Please input a value")
        continue
    if len(g) > 1:
        print("Please input a single value")
        continue
    if not g.isalpha():
        print("Please input a letter")
        continue
    gl = g.lower()
    if gl in seen:
        print(f"You've already guessed '{g}'")
        continue
    else:
        seen.add(gl)
    if gl not in indices:
        args.limit -= 1
        m = "Oops, '{}' is an invalid guess. {} guess{} remaining"
        m = m.format(g, args.limit, "" if args.limit == 1 else "es")
        print(m)
        continue
    ix = indices[gl]
    m = "You've guessed {} characters. {} guess{} remaining"
    m = m.format(len(ix), args.limit, "" if args.limit == 1 else "es")
    print(m)
    for i in ix:
        mask[i] = args.phrase[i]
    del indices[gl]

if indices:
    print("\nCroak... You lose!")
else:
    print("\nYou've cheated death this time. "
          "Next time, maybe not so lucky. Hehehe")

print(f"\nFinal answer: {args.phrase}")
