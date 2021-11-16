#!/usr/bin/env python3

import tkinter as tk

# This test program is a stand-in for ScrolledText, so I can examine
# the effects of 'yscrollcommand'.
#from tkinter import scrolledtext


root = tk.Tk()

frame = tk.Frame(root)
frame.pack(expand=True, fill="both")

text_out = tk.Text(frame, state=tk.DISABLED, wrap="char")
text_out.pack(side=tk.LEFT, expand=True, fill="both")

def from_scollbar_yview(*args, **kwargs):
    print("\ntext_out.yview: ", args, kwargs)
    return text_out.yview(*args, **kwargs)

scrollbar = tk.Scrollbar(frame, command=from_scollbar_yview)
scrollbar.pack(side=tk.LEFT, fill='y')

button = tk.Button(root, text="copy")
button.pack()

text_in = tk.Text(root)
text_in.pack(expand=True, fill="both")

def from_text_set(*args, **kwargs):
    print("\nscrollbar.set: ", args, kwargs)
    print("... and yview is: ", text_out.yview())
    return scrollbar.set(*args, **kwargs)

text_out.configure(yscrollcommand = from_text_set)

def on_click(event):
    y1 = text_out.yview()
    print("\ncopy start: ", y1)
    s = text_in.get("1.0","end-1c")

    # Attempted work-around. If the explanations below had been
    # correct this, though extremely hacky, would have worked.
    #yc = text_out.cget("yscrollcommand")
    #i = 0
    #def yc_ignore_n(*args):
    #    nonlocal i
    #    print("called yscrollcommand: ", i, *args)
    #    if i >= 1:
    #        print("setting yview[0]: ", y2[0])
    #        text_out.yview("moveto", y2[0])
    #        text_out.configure(yscrollcommand = yc)
    #        print("yscrollcommand restored!!")
    #    i += 1
    #    return True
    #if y1[0] > 0 or y1[1] < 1:
    #    print("implementing workaround")
    #    text_out.configure(yscrollcommand = yc_ignore_n)

    #tcl = tk.Tcl()
    #yc = text_out.cget("yscrollcommand")
    #def yc_wrapper(*args, **kwargs):
    #    ts = f"{yc} {args[0]} {args[1]}"
    #    print("yc wrapper: ", args, kwargs)
    #    print("yc wrapper command: ", ts)
    #    r = tcl.eval(ts)
    #    print("yc wrapper return: ", r)
    #    return r
    #text_out.configure(yscrollcommand = yc_wrapper)


    print(f"copying text: {len(s)} chars")
    text_out.configure(state=tk.NORMAL)
    # I now think these explanations are incorrect:
    #
    # If the text height changes, 'delete' will queue a
    # 'yscrollcommand' at the *front*. This *will* mess
    # with the 'yview'.
    text_out.delete(1.0, "end")
    # If the text height changes, 'insert' will queue a
    # 'yscrollcommand' at the *front* (before 'delete').
    text_out.insert("end-1c", s)
    text_out.configure(state=tk.DISABLED)

    print("setting yview[0]: ", 0.25)
    text_out.yview("moveto", 0.25)
    # When the contents only slightly exceed the size of the
    # text, 'yview' is set to an incorrect value. Why?
    y2 = text_out.yview()
    print("width: ", text_out.cget("width"))
    print("width: ", text_out.winfo_width())
    print("width: ", text_out.winfo_reqwidth())
    print("width: ", root.winfo_width())
    print("width: ", scrollbar.winfo_width())

    print("copy stop:", y2)

button.bind("<Button-1>", on_click)

# --- start program ---

root.mainloop()

"""
The first `yscrolledcommand` is queued for either state change (normal/disabled). The second from `insert` (not `delete`) is triggered when there is line wrapping *outside* the text view. Possibly to adjust the view position to any line wrapping. Since the adjustment causes the scrollbar to shrink, that implies it initially considers wrapped lines as a single line, then considers wrapped lines as multiple lines. Normally 25% is 25% but since the view can't move during the adjustment, and there's now more content, the top position percentage must be reduced. This is what causes the flickering.

For partially-obscured wrapped lines this only triggers at the bottom.

The initial configuration of the window view only considers wrapped lines that are visible and does not cause any flickering. For partially-obscured wrapped lines this only considers lines at the top. In their handling of wrapped lines, the initial and final view configurations are exact inverses.

Anyway I want to demonstrate the Tk is doing it completely wrong. Let's say I have 20 non-wrapped lines, a single lined wrapped into 20 lines, and 20 non-wrapped lines. I want to set the top of the scroll window at 25% which means the wrapped line will be completely visible.

At 25%, this is the first line that should be visible:

``` python
(12+20*2)*.25
>>> 13.0
```

However Tk shows a portion of the 11th line. What does this correspond to?

``` python
(1+20*2)*.25
>>> 10.25

10.25/(12+20*2)
>>> 0.1971153846153846
```

And that's what Tk adjusts to top of my `yview` to: roughy 19%.

Now let's say I have 40 non-wrapped lines and a single line wrapped into 20 lines. I want to set the top of the scroll window at 25% which means no wrapped lines will be visible.

I'd show the calculations, but they're pretty much exactly the same. The only difference is that the scrollbar will flicker. Instead:

``` lang-plaintext
copy start:  (0.0, 0.46153846153846156)
copying text: 1025 chars
setting yview[0]:  0.25
copy stop: (0.25, 0.8353658536585366)

scrollbar.set:  ('0.25', '0.8353658536585366') {}
... and yview is:  (0.25, 0.8353658536585366)

scrollbar.set:  ('0.1971153846153846', '0.6586538461538461') {}
... and yview is:  (0.1971153846153846, 0.6586538461538461)
```

So how do we solve this? You can try to undo Tk's math, working backwards to get the value that will be transformed into the value you want. However, you will still get flickering. The best solution is to wrap the lines yourself (`wrap = "none"`). For that, you need the window size. While window size is specified in terms of font size at creation, that is not maintained internally. All you can get is the window size in pixels using `winfo_width`. Luckily `Font.measure` provides a pixel width of a given string for the instantiated font.

The only question is it worth it? You'll have to bind to `<Configure>` to rewrap every time the widget is resized. That also means deciding how to adjust the scrollbar and which view to show.
"""
