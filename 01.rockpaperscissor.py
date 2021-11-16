#!/usr/bin/env python3

# Type: project
# Teaches: loops, conditionals, input
#
# Goal: As simple as possible. No exceptions, lists, functions, etc

import random


score_random = 0
score_user = 0

while True:
    limit = int(input("Play to which score limit (>0)? "))
    if limit > 0:
        break

while score_random < limit and score_user < limit:
    tie = "Tie! No points"
    win_user = "User wins!"
    win_random = "Computer wins!"

    rps = ["rock", "paper", "scissor"]
    choice_random = random.choice(rps)

    while True:
        choice_user = input("\nPlay which [rock, paper, scissor]? ")
        if (  choice_user == "rock"
           or choice_user == "paper"
           or choice_user == "scissor"
           ):
           break

    print("Computer picked: ", choice_random)

    if choice_random == "rock":
        if choice_user == "rock":
            print(tie)
        elif choice_user == "paper":
            print(win_user)
            score_user += 1
        elif choice_user == "scissor":
            print(win_random)
            score_random += 1
    elif choice_random == "paper":
        if choice_user == "rock":
            print(win_random)
            score_random += 1
        elif choice_user == "paper":
            print(tie)
        elif choice_user == "scissor":
            print(win_user)
            score_user += 1
    elif choice_random == "scissor":
        if choice_user == "rock":
            print(win_user)
            score_user += 1
        elif choice_user == "paper":
            print(win_random)
            score_random += 1
        elif choice_user == "scissor":
            print(tie)

    print("\nThe computer has: ", score_random, " points")
    print("The user has:     ", score_user, " points")

print("\nGame over: ", end="")

if score_user == score_random:
    print("both players are tied")
elif score_user < score_random:
    print("computer wins")
else:
    print("user wins")
