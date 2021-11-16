#!/usr/bin/env python3

# Type: project
# Teaches: modules, imports, functions

import morse_ui as mui
import morse_translate as mtr
import stats_translate as smtr


d2t = smtr.Duration2Text()

def duration2text(es):
    if not es:
        d2t.reset()
        return ""
    else:
        return d2t.update(es[len(d2t.onoff):])

mui.make_ui(mtr.text2dash, mtr.dash2text, duration2text)
