from nltk.parse.generate import generate
from nltk import CFG

import networkx as nx
from pnet import Pnet

MAX_DEPTH = 6
MAX_LENGTH = 5
GRAMMAR = '''
S -> 'S' A B | 's' C
A -> 'A' A C | 'a'
B -> 'b' | 'B' B A
C -> 'C' B | 'c' 
'''

grammar = CFG.fromstring(GRAMMAR)

sents = [s for s in generate(grammar, depth=MAX_DEPTH) if len(s) <= MAX_LENGTH]

p = Pnet(sents)

p01 = Pnet(['cd'])
p02 = Pnet(['ab'])
p1 = Pnet(['ab', 'cd'])
p2 = nx.relabel_nodes(p1,{1:2})


