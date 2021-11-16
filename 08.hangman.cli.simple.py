#!/usr/bin/env python3

# Type: project
# Teaches: algorithms, loops, sets

import random


words = [ "one", "two", "three"
        , "four", "five", "six"
        , "seven", "eight", "nine"
        ]

# Sample 'n' words from the pool to create a random phrase.
phrase = " ".join(random.sample(words, 2))

# The guess limit.
limit = 10

# For the given phrase map each unique character (converted
# to lowercase) to a list of indices at which that character
# appears. Once a character has been guessed, remove it from
# the mapping. The mapping will be empty when all characters
# have been guessed.
indices = {}

for i,c in enumerate(phrase):
    c = c.lower()
    if c.isspace():
        continue
    if c not in indices:
        indices[c] = []
    indices[c].append(i)

# For the given phrase create a mask to present to the user.
# Alphabetic characters are replaced with underscores while
# all other characters are presented unchanged.
mask = []

for c in phrase:
    if c.isalpha():
        mask.append("_")
    else:
        mask.append(c)

# Store every guessed character to avoid penalizing repeated
# guesses.
seen = set()

m = f"Let's play hangman! {limit} guesses left"
print(m, end="\n\n")

# While there are guesses left and the phrase hasn't been
# fully revealed, loop over the game logic. The guess limit
# counts downwards. Remember that of any numeric type only
# zero is False.
while limit and indices:
    print("Hangman says: ", "".join(mask))
    # Collect user input. Invalid input continues the loop,
    # skipping everything below, so we end up right back here.
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
    # For guessing characters, case doesn't matter. Convert
    # everything to the same case to facilitate comparisons.
    gl = g.lower()
    # Has the character been guessed previously? If so don't
    # penalize the user but continue the loop to get another
    # guess. Otherwise add it to the list of seen characters.
    if gl in seen:
        print(f"You've already guessed '{g}'")
        continue
    else:
        seen.add(gl)
    # If the guess isn't valid, decrement the guess counter
    # and continue the game loop.
    if gl not in indices:
        limit -= 1
        m = "Oops, '{}' is an invalid guess. {} guess remaining"
        m = m.format(g, limit)
        print(m)
        continue
    # To reach here, the guess must have been correct. Use the mapping
    # to get every index containing the guessed character. Since case
    # matters for presentation avoid filling every index with the input
    # guess. Rather, copy the character from the corresponding index of
    # the original phrase.
    ix = indices[gl]
    m = "You've guessed {} characters. {} guess remaining"
    m = m.format(len(ix), limit)
    print(m)
    for i in ix:
        mask[i] = phrase[i]
    # Delete the now-unnecessary character from the mapping. When
    # all characters are guessed the mapping will be empty. This is
    # why it is important to store previous guesses as otherwise a
    # new guess of the same character would register as incorrect.
    del indices[gl]
    # The new mask will be displayed at the start of the next loop.

# The game is over - determine win/lose.
if indices:
    print("\nCroak... You lose!")
else:
    print("\nYou've cheated death this time. "
          "Next time, maybe not so lucky. Hehehe")

print("\nFinal answer: ", phrase)
