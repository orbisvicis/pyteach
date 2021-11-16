#!/usr/bin/env python3

# Type: project
# Teaches: dictionaries, comprehensions

import math


letter2dash =\
    { "a": ".-"
    , "b": "-..."
    , "c": "-.-."
    , "d": "-.."
    , "e": "."
    , "f": "..-."
    , "g": "--."
    , "h": "...."
    , "i": ".."
    , "j": ".---"
    , "k": "-.-"
    , "l": ".-.."
    , "m": "--"
    , "n": "-."
    , "o": "---"
    , "p": ".--."
    , "q": "--.-"
    , "r": ".-."
    , "s": "..."
    , "t": "-"
    , "u": "..-"
    , "v": "...-"
    , "w": ".--"
    , "x": "-..-"
    , "y": "-.--"
    , "z": "--.."
    , "1": ".----"
    , "2": "..---"
    , "3": "...--"
    , "4": "....-"
    , "5": "....."
    , "6": "-...."
    , "7": "--..."
    , "8": "---.."
    , "9": "----."
    , "0": "-----"
    }

dash2letter = {v:k for k,v in letter2dash.items()}

# Source: http://norvig.com/mayzner.html
letter_frequencies =\
    { "e": 12.49/100
    , "t":  9.28/100
    , "a":  8.04/100
    , "o":  7.64/100
    , "i":  7.57/100
    , "n":  7.23/100
    , "s":  6.51/100
    , "r":  6.28/100
    , "h":  5.05/100
    , "l":  4.07/100
    , "d":  3.82/100
    , "c":  3.34/100
    , "u":  2.73/100
    , "m":  2.51/100
    , "f":  2.40/100
    , "p":  2.14/100
    , "g":  1.87/100
    , "w":  1.68/100
    , "y":  1.66/100
    , "b":  1.48/100
    , "v":  1.05/100
    , "k":  0.54/100
    , "x":  0.23/100
    , "j":  0.16/100
    , "q":  0.12/100
    , "z":  0.09/100
    }

# For the english language, what is the average length of a morse code
# (letter)? Exclude spacing codes and not considering timing or duration.
avg_code_length =\
    sum(letter_frequencies[k]*len(v)
            for k,v in letter2dash.items()
            if k in letter_frequencies
       )

# Source: http://norvig.com/mayzner.html
avg_word_length = 4.79

# For the english language, what are the frequencies for each possible morse
# code duration? Include spacing codes and assume a large sample size, so that
# given a word count 'n' the space to word ratio is: lim(n->inf)[(n-1)/n] = 1
#
# For example, given an average word length of 4 and an average code length of
# 2, the resulting average word would be composed of the following basic units:
#   . .|. .|. .|. ._
# Where:
#   period      -> letter unit as seen in 'letter_frequencies'
#   space       -> intra-letter gap, duration 1
#   pipe        -> inter-letter gap, duration 3
#   underscore  -> inter-word gap, duration 7
#
# The size of each unit is as given:
#
#   period      -> (avg word length) * (avg code length)
#   space       -> (avg code length - 1) * (avg word length)
#   pipe        -> (avg word length - 1)
#
# The number of underscore units (inter-word gaps) depends on the the total
# number of units 'n' and approaches infinity as 'n' increases. It is given
# with the following relations:
#   n + 1 = Z * (WL + 1)
#   n = Z * (WL + Y)
# Where:
#   n   -> total number of units
#   Z   -> number of average words
#   WL  -> length of an average word excluding the trailing underscore unit
#   Y   -> number of underscore units per average word, 0 <= Y <= 1
#
# Where WL can be expanded to its constituent units given:
#   w   -> average word length
#   c   -> average code length
#
# Notice that when Y is 1, an extra unit is produced.
# 
# Solving for Z:
#   Z[w*c + (w-1) + ((c-1)*w) + 1] = n + 1
#   Z[w*c + w - 1 + w*c - w + 1] = n + 1
#   Z[w*c + w*c] = n + 1
#   Z[2wc] = n+1
#   Z = (n+1) / (2wc)
#
# Solving for Y by substituting Z:
#   Z[w*c + w - 1 + w*c - w + Y] = n
#   Y = n/Z - 2wc + 1
#   Y = (n)*(2wc)/(n+1) - 2wc + 1
#   Y = 2wcn/(n+1) - 2wc*(n+1)/(n+1) + 1
#   Y = (2wcn - 2wc*(n+1))/(n+1) + 1
#   Y = (2wcn - 2wcn -2wc)/(n+1) + 1
#   Y = -2wc/(n+1) + 1
#
# This can be understood from another perspective:
#   n = Z*WL + (Z-1)
#   Y = (Z-1)/Z
#
# Note that Y becomes negative when Z < 1. In such a case Y should be set to
# zero.
#
# Once the number of basic on/off units (1,3,7) in an average word is known it
# is possible to calculate the frequency of each type of unit. The multi-unit
# letter frequencies, as seen in 'letter_frequencies', distribute across the
# constituent units, for example:
#   r freq  -> 0.0628
#   r       -> .-.
#   .       -> 0.0628 / 3 * 2
#   -       -> 0.0628 / 3 * 1

# The resulting letter-unit frequencies must also be adjusted for non-letter
# frequencies where:
#   LUF = LUF * (LU / U)
# Where:
#   LUF     -> letter-unit frequencies
#   LU      -> number of letter units per average word
#   U       -> total number of units per average word
#
# After which the non-letter frequencies can be added.
def time_frequencies(n=math.inf):
    units_letters = avg_word_length * avg_code_length
    units_gaps = avg_word_length - 1
    units_letter_gaps = (avg_code_length - 1) * avg_word_length

    avg_words = (n+1) / (2 * avg_word_length * avg_code_length)
    units_word_gap = ((-2 * avg_word_length * avg_code_length) / (n+1)) + 1
    units_word_gap = max(0, units_word_gap)

    units = units_letters + units_gaps + units_letter_gaps + units_word_gap

    char2time =\
        { ".": 1
        , "-": 3
        }

    unit_frequencies =\
        { 1: 0
        , 3: 0
        , 7: 0
        }

    for letter,freq in letter_frequencies.items():
        try:
            code = letter2dash[letter]
        except ValueError:
            continue
        p = freq / len(code) * (units_letters / units)
        for c in code:
            unit_frequencies[char2time[c]] += p

    unit_frequencies[1] += units_letter_gaps / units
    unit_frequencies[3] += units_gaps / units
    unit_frequencies[7] += units_word_gap / units

    return (unit_frequencies, avg_words)
