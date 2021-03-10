from nltk.parse.generate import generate
from nltk import CFG
from networkx.readwrite import json_graph

import json
import random as rnd
import networkx as nx
from pnet import Pnet
from restore import *

MAX_DEPTH = 8
MAX_LENGTH = 20
GRAMMAR = '''
S -> '>' F
F -> '*' V V | 'f' V
V -> 'x' | '(' F ')' 
'''
# GRAMMAR = '''
# S -> 'S' A B | 's' C
# A -> 'A' A C | 'a'
# B -> 'b' | 'B' B A
# C -> 'C' B | 'c' 
# '''

grammar = CFG.fromstring(GRAMMAR)

sents = [s for s in generate(grammar, depth=MAX_DEPTH) if len(s) <= MAX_LENGTH]

# res = restore(sents, maxt=8, maxh=8)
# minres = min(res, key=len)


# for i in range(1, len(sents)):
# 	print(f'Attempt with sample size {i}')
# 	curr_sents = rnd.sample(sents, i)
# 	res = restore(curr_sents)
# 	minres = min(res, key=len)

# 	if all(check_grammar(minres, s) for s in sents):
# 		print(f'\n\nSuccess at i={i}\n')
# 		for s in curr_sents:
# 			print(''.join(s))
# 		break

sents = [
	'>fx',
	'>f(fx)',
	'>f(f(fx))',
	'>f(f(*xx))',
	'>f(*x(fx))',
	'>f(*xx)',
	'>*(*x(fx))(*xx)',
	'>*(*xx)(*(fx)x)',
	'>*(*(fx)(fx))(*x(fx))',
	'>*(*(fx)x)(*(fx)(fx))',
	'>*(*(*xx)(fx))(*x(*xx))',
	'>*(*(*xx)x)(*(fx)(*xx))',
	'>*(fx)(fx)',
	'>*(fx)(f(fx))',
	'>*(fx)(*xx)',
	'>*(fx)(*x(fx))',
	'>*(fx)x',
	'>*(f(fx))(*(*xx)x)',
	'>*xx',
	'>*x(*x(fx))',
	'>*x(*xx)',
	'>*x(f(fx))',
	'>*x(fx)',
]
# sents = [
# 	'#',
# 	'aaca',
# 	'aabaca',
# 	'a(a)ca',
# 	'aab(a)ca',
# 	'aababaca',
# 	'a((a))ca',
# 	'a(a)baca',
# 	'aab(a)baca',
# 	'a((a))baca',
# 	'a(a)b(a)ca',
# 	'a(a)babaca',
# 	'a((a))b(a)ca',
# 	'a((a))babaca',
# 	'a(a)b(a)baca',
# 	'a((((a))))baca',
# 	'a((a))b(a)baca',
# 	'a((((a))))b(a)ca',
# 	'a((a))babababaca'
# ]

# sents = [
# 	'>*(fx)(*xx)',
# 	'>*(fx)(*x(fx))',
# 	'>*(fx)(fx)',
# 	'>*(fx)(f(fx))',
# 	'>*(fx)x',
# 	'>*x(*xx)',
# 	'>*x(*x(fx))',
# 	'>*x(fx)',
# 	'>*x(f(fx))',
# 	'>*xx',
# 	'>f(*xx)',
# 	'>f(*x(fx))',
# 	'>f(fx)',
# 	'>f(f(fx))',
# 	'>fx'
# ]

# p = Pnet.load('graph.json')
# p1 = Pnet(['fae'])
# p2= Pnet(['faac', 'fbbc'])
# p2.factorize((0,-1))

# should be 3 3


# p = Pnet(sents)

# algo(p,3,15)
# S, gram = net_to_grammar(p,3)
# print(gram)
# check_grammar(gram, sents)
# # p.factorize((1,-1))

# p.divide((29,3), t=3,h=7)
# p.divide((4,3), t=3,h=7)
# p.divide((23,3), t=3,h=7)

# p.divide((1,3),t=3,h=7)

# p.factorize((7,2))


		




# p01 = Pnet(['cd'])
# p02 = Pnet(['ab'])
# p1 = Pnet(['ab', 'cd'])
# p2 = nx.relabel_nodes(p1,{1:2})



# p.add_sent('E',9,11)
# p.add_sent('I',0,8)
# p.add_sent('i',0,1)
# p.add_sent('I',9)
# p.add_sent('qq')
# p.add_sent('qq',5)

# pcut = p.subcopy(9)

# # leads to visual bug
# pp = p.compose(pcut, 5)