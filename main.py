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



p.add_sent('E',9,11)
p.add_sent('I',0,8)
p.add_sent('i',0,1)
p.add_sent('I',9)
p.add_sent('qq')
p.add_sent('qq',5)

pcut = p.subcopy(9)

# leads to visual bug
pp = p.compose(pcut, 5)