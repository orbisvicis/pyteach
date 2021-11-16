#!/usr/bin/env python3

import stats
import morse_data as mdt


class Duration2Text:
    # Average morse code length is 2.5 codes per english character
    # excluding spacing and not considering timings. The average
    # english word is 5.1 characters. Set the window minimum (in
    # codes) to a multiple of the average word, including spacing
    # codes which doubles the count.
    window_min = int(mdt.avg_code_length * 2 * 5.1 * 1)
    # Set the window maximum (in codes) based on the minimum. The
    # start of the next window will lie between these two values.
    window_max = int(window_min * 2)

    def __init__(self):
        self.onoff = []
        self.codes = []
        self.seen = set()
        self.text = []

        self.index = 0
        self.index_text = 0

    def reset(self):
        self.onoff.clear()
        self.codes.clear()
        self.seen.clear()
        self.text.clear()

        self.index = 0
        self.index_text = 0

    def update(self, onoffs, append_only=True):
        # Should not exceed window_max when adding
        while onoffs:
            i = self.window_max - len(self.onoff) + self.index
            self.onoff.extend(onoffs[:i])
            self.decode()
            onoffs = onoffs[i:]
        return "".join(t[0] for t in self.text)

    def decode(self):
        # Keys from the current analysis window
        ks = self.onoff[self.index:]
        # Can't do anything without data
        if not ks:
            return
        # Duration-only from the same window
        ds = [k[2] for k in ks]
        # Classified/categorized data
        cs = []
        # Possible break lists, and cluster lists for keys and durations
        bss = []
        dcss = []
        kcss = []
        # Determine the maximum number of clusters
        counts = {}
        for k in ks:
            if k[2] in counts:
                counts[k[2]].append(k)
            else:
                counts[k[2]] = [k]
            if len(counts) >= 3:
                break
        # Determine all possible clusterings based on groups already seen
        for i in range(max(len(self.seen), 1), len(counts) + 1):
            if i == 1:
                bss.append([min(ds), max(ds)])
                dcss.append([ds])
                kcss.append([ks])
                continue
            if i == len(counts) and sum(map(len, counts.values())) == len(ks):
                cks = sorted(counts)
                bss.append([cks[0]] + cks)
                dcss.append([[d]*len(counts[d]) for d in cks])
                kcss.append([counts[d] for d in cks])
                continue
            bs = stats.jenks_breaks(ds, i)
            dcs = stats.break_list(ds, bs)
            kcs = stats.break_list(ks, bs, key=lambda l:l[2])
            bss.append(bs)
            dcss.append(dcs)
            kcss.append(kcs)
        # Rank each possible clustering based on relative expected means
        rs = stats.all_clusters_em_var(dcss, [1,3,7], collapse=False)
        # Remove rankings that don't include elements already seen
        rs = [r for r in rs if
                len(self.seen) > len(r[2]) or self.seen <= set(r[2])
             ]
        # Determine if any of the rankings produce an impossible result.
        # '7' is reserved for off-keying. A '7' keyed on is impossible.
        rc = [0,0]
        for r in rs:
            if 7 not in r[2]:
                s = True
            elif any(i[0] for i in kcss[r[0]][-1]):
                s = False
            else:
                s = True
            r.append(s)
            rc[s] += 1
        # Select the best ranking
        r = min\
            ( (r for r in rs if r[3] or not rc[True])
            , key=lambda l: l[1]
            )
        # Default to '1' for a single cluster
        if not r[2]:
            r[2] = (1,)
        # Classify the data into on-off state and category
        for k in ks:
            for ib,b in enumerate(bss[r[0]][1:]):
                if k[2] <= b:
                    s = k[0]
                    c = r[2][ib]
                    cs.append([k[0], c])
                    break
            else:
                raise ValueError
        # update codes
        self.codes[self.index:] = cs
        # Update seen groups for the current analysis window
        self.seen = set(r[2])
        # convert codes to [char,index_start,index_stop]. The closing index
        # includes the closing delimiter as that is an important component.
        ic1 = self.text[self.index_text][1] if self.text else 0
        cc = ""
        self.text[self.index_text:] = []
        for ic2,c in enumerate(self.codes[ic1:], start=ic1):
            if c[0]:
                if c[1] == 1:
                    cc += "."
                elif c[1] == 3:
                    cc += "-"
                else:
                    cc += "?"
            else:
                if c[1] == 1:
                    continue
                if cc:
                    ch = mdt.dash2letter.get(cc, "?")
                    self.text.append([ch, ic1, ic2])
                    cc = ""
                    ic1 = ic2 + 1
                if c[1] == 7:
                    self.text.append([" ", ic2, ic2])
        if cc:
            ch = mdt.dash2letter.get(cc, "?")
            self.text.append([ch, ic1, None])
        # Collect some statistics
        finals = {}
        fbreak = None
        for ic,c in enumerate(cs[-1:self.window_min-1:-1], -len(cs)+1):
            ic *= -1
            if c[1] not in finals:
                finals[c[1]] = ic
                fbreak = None
            elif not c[0] and c[1] > 1:
                fbreak = ic
            if len(finals) >= len(self.seen) and fbreak is not None:
                break
        # Minimize the overlap (reduce work) between windows such that
        # * new window must start after old window_min
        # * maximize the number of clusters for the new window
        # * avoids character breaks (put the word break with the old window)
        move = (   (len(cs) > self.window_min)
               and (  (len(finals) >= 3 and fbreak is not None)
                   or (len(cs) >= self.window_max)
                   )
               )
        # Move the analysis window and update the text pointer
        if move:
            self.index +=\
                ( fbreak+1
                    if fbreak is not None else
                  min(finals.values())
                )
            self.seen = set(finals.keys())
            it = max(len(self.text) - 1, 0)
            while it > self.index_text:
                t = self.text[it]
                if t[2] is not None and t[2] < self.index:
                    break
                it -= 1
            self.index_text = it
        # Done
