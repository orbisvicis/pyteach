#!/usr/bin/env python3

# Type: project
# Teaches: dictionaries, functions, algorithms

import morse_data as mdt


space_char = 3
space_word = 7

def text2dash(s):
    return (" "*space_word).join(
        (" "*space_char).join(mdt.letter2dash.get(c.lower(),"?") for c in w)
        for w in s.split()
    )

def dash2text(s):
    i = 0
    l = []
    for j in range(1, len(s)+1):
        if j < len(s) and s[j].isspace() == s[j-1].isspace():
            continue
        p = s[i:j]
        i = j
        if p.isspace() and len(p) <= space_char:
            continue
        if p.isspace():
            l.append(" ")
        else:
            l.append(mdt.dash2letter.get(p, "?"))
    return "".join(l)
