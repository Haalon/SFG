from nltk.parse.generate import generate
from nltk import CFG

import networkx as nx
from pnet import Pnet

MAX_DEPTH = 6
MAX_LENGTH = 11
GRAMMAR = '''
S -> 'S' A B | 's' C
A -> 'A' A C | 'a'
B -> 'b' | 'B' B A
C -> 'C' B | 'c' 
'''

grammar = CFG.fromstring(GRAMMAR)

sents = [s for s in generate(grammar, depth=MAX_DEPTH) if len(s) <= MAX_LENGTH]

p = Pnet(sents)
