# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:37:34 2018

@author: FUBAR1342
"""

from random import shuffle
import time

class card:
    def __init__(self, suit, rank):
        self.suit = suit.lower()  # make sure that the suit is spelled in all lowercase characters
        self.rank = rank

    def isBlackOrRed(self):
        # Function to check if a card is black or red
        # Hearts and Diamonds are red cards, all other are black cards
        if self.suit == "hearts" or self.suit == "diamonds":
            return "red"
        else:
            return "black"

    def isFaceCard(self):
        # Function to check if a card is a face card
                
        faceCards = [11, 12, 13]
        for faceCard in faceCards:
            if self.rank == faceCard:
                return True
        return False

    def getDescription(self):
        # Returns a two item list
        return [self.rank, self.suit]

    def __repr__(self):
        # Returns a nice explanation of the card
        # for example: "ace of diamonds"
        rank = self.rank
        if rank == 1:
            rank = "ace"
        elif rank == 11:
            rank = "jack"
        elif rank == 12:
            rank = "queen"
        elif rank == 13:
            rank = "king"

        return "{} of {}".format(rank, self.suit)

#############

class cardStack:
    # Class for both decks and hands
    def __init__(self, name=""):
        # Optional name variable, by default it's empty
        self.stack = []
        self.name = name

    def giveFullDeck(self):
        # Gives the stack a full set of cards (52 cards)
        suits = ["clubs", "diamonds", "hearts", "spades"]
        for suit in suits:
            for rank in range(1, 14):
                self.stack.append(card(suit, rank))  # Uses the card class to create cards

    def stackSize(self):
        # Return the length of the stack
        return len(self.stack)

    def shuffle(self, amount=1):
        # Shuffle the stack, just for fun there's an optional
        # parameter, for how often you want to shuffle the stack
        # adds "realism" (default is one time)
        for i in range(0, amount):
            shuffle(self.stack)

    def deal(self, amount, stack, position=-1):
        # deal a card to another stack, optional
        # parameter for where to deal the card, default is
        # at the end of the stack (-1)
        for i in range(0, amount):
            dealt = self.stack[0:1]
            self.stack = self.stack[1:]
            for i in dealt:
                stack.stack.insert(position, i)

    def splitDeckIntoTwoStacks(self, stack1, stack2):
        # Split the stack into two other stacks
        # Will try to split it 50/50
        while len(self.stack) > 0:
                self.deal(1, stack1)
                self.deal(1, stack2)

    def splitStack(self, stack):
        # Does a similar thing to "splitDeckIntoTwoStacks"
        # but this just splits itself into one other stack
        while len(self.stack) > len(self.stack)/2:
                self.deal(1, stack)

    def __str__(self):
        spaces = ""
        print(self.name)
        for card in self.stack:
            print(spaces, card)
            spaces += "  "
        return ""

def highPlay(mainDeck):
    play1 = deck.stack[0].rank
    print("          You played:", deck.stack[0])
    play2 = deck.stack[1].rank
    print("Your opponent played:", deck.stack[1])
    if play1 > play2:
        return "play1"
    elif play2 > play1:
        return "play2"
    else:
        return "draw"

#  Lay out the things befoe the game
deck = cardStack()  # Create a stack to be the main deck
deck.giveFullDeck()  # give this stack a full deck of cards
deck.shuffle(amount=5)  # shuffle this full deck of cards
player1Hand = cardStack()  # give player1Hand a stack (empty)
player2Hand = cardStack()  # Same for player2Hand
deck.splitDeckIntoTwoStacks(player1Hand, player2Hand) # Split the full deck of cards to player1Hand and player2Hand (26 cards)
round = 0


# Explain the rules for the game
print("""             W E L C O M E  T O  W A R !
    The rules are simple, play a card and pray to god that it is a
    higher rank than your opponents card.
    If you play the same card rank as your opponent, it's WAR
    You and your opponent will play two cards, only the second one
    needs to be higher rank than your opponents second card
    if it is, you get all the cards played that round
                L E T ' S  B E G I N !""")

wait = input("                 press ENTER to start")  # all "wait" variables, are just there to wait for the player to input something

# main game loop
while True:
    # shuffle the players' deck at the start of each round
    player1Hand.shuffle()
    player2Hand.shuffle()
    round += 1  # add to round timer by 1
    print()
    print("-----------------------------------------")
    print()
    print("this is round:", round)  # tell the player what round it is
    print("and you have", len(player1Hand.stack), "cards")  # tell the player how many cards he has
    deck.stack = []  # empty the main deck stack, should in theory always be empty at the start of a round
                     # but to be safe, it's forced emptied here
    wait = input("press ENTER to play a card")
    print()

    # loop for the rounds
    player1Hand.deal(1, deck)  # Deal one card into the deck
    player2Hand.deal(1, deck)  # Deal one card into the deck
    while True:
        highestPlay = highPlay(deck)  # Get who played the highest card, or if it was a draw
        # Give the cards to the player that played the highest rank or
        # start the code for a draw
        if highestPlay == "play1":
            print("YOU WON THE ROUND!")
            print("You won the following cards:")
            print(deck)
            deck.deal(len(deck.stack), player1Hand)
            break
        elif highestPlay == "play2":
            print("Sadly, your opponent won the round")
            print("He won the following cards:")
            print(deck)
            deck.deal(len(deck.stack), player2Hand)
            break

        elif highestPlay == "draw":
            print("AND IT'S A DRAW!")
            wait = input("press ENTER to play a card facedown")
            player1Hand.deal(1, deck)  # deal the card to the back of the deck list (default)
            wait = input("press ENTER to play a card faceup")
            print()
            player1Hand.deal(1, deck, 0)  # deal the card to the front of the deck list
            player2Hand.deal(1, deck)
            player2Hand.deal(1, deck, 0)
            print("Your opponent played two cards")

    if len(player1Hand.stack) == 0:  # if player1Hand.stack has no cards in it, player1Hand lost the game
        print("Sadly, you lost the war")
        wait = input("the game is over, press ENTER to quit")
        break
    elif len(player2Hand.stack) == 0:  # if player2Hand.stack has no cards in it, player1Hand won the game
        print("YOU ANNHILATED YOUR OPPONENT!")
        wait = input("the game is over, press ENTER to quit")
        break