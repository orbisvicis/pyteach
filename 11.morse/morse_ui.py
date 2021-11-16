#!/usr/bin/env python3

import types

import tkinter as tk

from tkinter import ttk
from tkinter import scrolledtext
from tkinter import font


class FilterAutoRepeat:
    press = tk.EventType.KeyPress
    release = tk.EventType.KeyRelease

    def __init__(self):
        self.db = {}
        self.i = 0

    @staticmethod
    def event_args(event):
        exclude =\
            { "num", "keycode", "keysym", "keysym_num"
            , "char", "send_event", "type", "widget"
            , "x_root", "y_root", "delta"
            }
        args =\
            {k:v for k,v in vars(event).items()
             if k not in exclude and v != "??"
            }
        return args

    @staticmethod
    def repeated(n1, n2):
        return\
            (   n1 and n2
            and n1.event.type != n2.event.type
            and n1.event.time == n2.event.time
            and n1.event.serial == n2.event.serial
            )

    def on_key(self, event, window):
        def after():
            h[-1] = None
            if self.repeated(node, node.next):
                return
            window.event_generate\
                ( f"<<PhysicalKeyRelease-{event.keysym}>>"
                , **self.event_args(node.event)
                )
        try:
            h = self.db[event.keysym]
        except KeyError:
            h = [None, None]
            self.db[event.keysym] = h
        else:
            h[0].prev = None
        node = types.SimpleNamespace(event=event, prev=h[0], next=None)
        if node.prev:
            node.prev.next = node
        h[0] = node
        if h[-1]:
            cb_id, cb = h[-1]
            cb()
            window.after_cancel(cb_id)
        if event.type == self.press and not self.repeated(h[0], h[0].prev):
            window.event_generate\
                ( f"<<PhysicalKeyPress-{event.keysym}>>"
                , **self.event_args(event)
                )
        elif event.type == self.release:
            h[-1] = (window.after(0, after), after)


class OnOffButton(tk.Label):
    def __init__(self, *args, pressed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = pressed
        self.configs = [self.config_get(), self.config_get()]
        self.configs_set = [False, False]

        name_parent = type(self).__bases__[0].__name__
        name_class = type(self).__name__
        tags = list(self.bindtags())
        index = tags.index(name_parent)
        tags[index:index] = [name_class]
        self.bindtags(tags)

    def config_get(self):
        return {k:v[4] for k,v in self.config().items() if len(v) >= 5}

    def config_unalias(self, config):
        a = { k:v[-1].lstrip("-") for k,v in self.config().items()
              if len(v) < 5
            }
        c = config.copy()
        for alias,full in a.items():
            if alias not in c:
                continue
            c[full] = c[alias]
            del c[alias]
        return c

    def config_is_set(self, pressed=None):
        if pressed is None:
            pressed = self.pressed
        return self.configs_set[pressed]

    def config_update(self, c, pressed=None, replace=False):
        if pressed is None:
            pressed = self.pressed
        config = self.configs[pressed]
        if config is None:
            config = self.configs[pressed] = {}
        if replace:
            config.clear()
        config.update(c)
        self.configs_set[pressed] = True

    def config_save(self):
        c = self.config_get()
        if  (   not self.configs_set[self.pressed]
            and self.configs[self.pressed] != c
            ):
            self.configs_set[self.pressed] = True
        self.configs[self.pressed] = c

    def config_restore(self):
        c = self.configs[self.pressed]
        if not c:
            return
        self.config(c)

    def press(self, event):
        if self.pressed:
            return
        self.config_save()
        self.pressed = not self.pressed
        self.config_restore()
        self.event_generate("<<Press>>", time=event.time)

    def release(self, event):
        if not self.pressed:
            return
        self.config_save()
        self.pressed = not self.pressed
        self.config_restore()
        self.event_generate("<<Release>>", time=event.time)


class RecordOnOff:
    def __init__(self, on_add=None, on_reset=None):
        self.entries = []
        self.current = None
        self.on_add = on_add
        self.on_reset = on_reset

    def record(self, event, press):
        if not self.current and not press:
            return (False,)
        if not self.current:
            self.current = (press, event.time)
            return (True,)
        if self.current[0] == press or self.current[1] >= event.time:
            return (False,)
        entry = self.current + (event.time - self.current[1],)
        self.current = (press, event.time)
        return self.add(entry)

    def add(self, entry):
        self.entries.append(entry)
        if self.on_add:
            return (True, self.on_add(self))
        return (True,)

    def reset(self):
        c = bool(self.entries or self.current)
        if c:
            self.entries.clear()
            self.current = None
        if self.on_reset:
            return (c, self.on_reset(self, c))
        return (c,)


def text_disabled_set(text, s):
    text.configure(state=tk.NORMAL)
    text.delete(1.0, "end")
    text.insert("end-1c", s)
    text.configure(state=tk.DISABLED)

def text_on_modify(event, text, translator):
    if not event.widget.edit_modified():
        return
    s = translator(event.widget.get("1.0","end-1c"))
    text_disabled_set(text, s)
    event.widget.edit_modified(False)

def make_ui(text2dash=str, dash2text=str, duration2text=str):
    root = tk.Tk()
    root.title("Morse Manipulator")

    text_font = font.nametofont("TkFixedFont")
    text_font.configure(family="DejaVu Sans Mono", size=12)

    text_font = font.nametofont("TkDefaultFont")
    text_font.configure(family="DejaVu Sans", size=12)

    note = ttk.Notebook(root)
    note.enable_traversal()
    note.pack(fill=tk.BOTH, expand=True)

    tabs =\
        [ ("text → dot/dash", "text2dash", text2dash)
        , ("dot/dash → text", "dash2text", dash2text)
        ]

    for title,name,translator in tabs:
        frame = tk.Frame(note, padx=4, pady=4, name=name)

        note.add(frame, text=title)

        text_out = scrolledtext.ScrolledText\
            ( frame
            , wrap=(tk.WORD if name == "dash2text" else tk.CHAR)
            , width=100, height=15
            , padx=10, pady=10
            , state=tk.DISABLED
            )
        text_out.pack(fill=tk.BOTH, expand=True, pady=(0,4))

        text_in = scrolledtext.ScrolledText\
            ( frame
            , wrap=(tk.WORD if name == "text2dash" else tk.CHAR)
            , width=100, height=15
            , padx=10, pady=10
            )
        text_in.pack(fill=tk.BOTH, expand=True)

        text_in.bind\
            ( "<<Modified>>"
            , lambda ev,tx=text_out,tr=translator:
                text_on_modify(ev,tx,tr)
            )

    frame = tk.Frame(note, padx=4, pady=4, name="duration2text")

    note.add(frame, text="duration → text")

    text_out = scrolledtext.ScrolledText\
        ( frame
        , width=100, height=15
        , padx=10, pady=10
        , state=tk.DISABLED
        )
    text_out.pack(fill=tk.BOTH, expand=True, pady=(0,4))

    frame_outer = tk.Frame(frame, bg="white")
    frame_outer.pack(fill=tk.BOTH, expand=True)

    frame_outer.rowconfigure([0], weight=1)
    frame_outer.columnconfigure([0,1], weight=1)

    label = OnOffButton\
        ( frame_outer
        , text="<spacebar|return|click: off>"
        , padx=60, pady=30
        , border=6, relief=tk.RAISED
        )
    label.grid(row=0, column=0, sticky="e", padx=(0,10))
    label.config({"background":"#b2ff94"})
    label.config_update\
        ( c=
            {"bg":"#ff3800", "relief": tk.SUNKEN
            , "text": "<spacebar|return|click: on>"
            }
        , pressed=True
        )

    def handle_recording(record, text, translator):
        text_disabled_set(text, translator(record.entries))

    def return_root(e):
        if note.index("current") != note.index("end") - 1:
            return
        far.on_key(e, label)

    def record_cb(record, change=True):
        if not change:
            return
        label.after(0, handle_recording, record, text_out, duration2text)

    far = FilterAutoRepeat()
    rec = RecordOnOff(record_cb, record_cb)

    button = tk.Button\
        ( frame_outer
        , text="reset"
        , padx=60, pady=30-1
        , border=6, highlightthickness=0
        , bg="#ffff94", activebackground="#ffffbd"
        , command=rec.reset
        )
    button.grid(row=0, column=1, sticky="w", padx=(10,0))

    root.bind("<KeyPress-Return>", return_root)
    root.bind("<KeyRelease-Return>", return_root)
    root.bind("<KeyPress-space>", return_root)
    root.bind("<KeyRelease-space>", return_root)

    label.bind("<<PhysicalKeyPress-Return>>", label.press)
    label.bind("<<PhysicalKeyRelease-Return>>", label.release)
    label.bind("<<PhysicalKeyPress-space>>", label.press)
    label.bind("<<PhysicalKeyRelease-space>>", label.release)
    label.bind("<ButtonPress-1>", label.press)
    label.bind("<ButtonRelease-1>", label.release)

    label.bind("<<Press>>", lambda e: rec.record(e, True))
    label.bind("<<Release>>", lambda e: rec.record(e, False))

    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    root.mainloop()
