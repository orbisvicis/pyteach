#!/usr/bin/env python3

"""
A helper to generate the word list required by '09.random.words.py'
"""

import argparse

import requests
import bs4

parser = argparse.ArgumentParser(description="Extract HTML text to file")
parser.add_argument("file", help="destination file (must not exist)")
parser.add_argument("url", help="fetch from this resource")

args = parser.parse_args()


r = requests.get(args.url)
s = bs4.BeautifulSoup(r.text, "html.parser")
t = " ".join(s.stripped_strings)

try:
    f = open(args.file, "x")
except FileExistsError as e:
    parser.error(str(e))

with f:
    f.write(t)
