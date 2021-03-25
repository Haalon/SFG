# SFG - Separated grammar research package

This package is centered around separated grammars and their inference/induction/restoration

Grammar inference is a process of producing a formal grammar from a set of given sentences.
Implemented inference algorithm is based on prefix-parallel-series network analysis and transformation.

Context-Free grammar is separated if all its productions start with a terminal
And for all nonterminals, no two productions start with the same terminal

Example of a separated grammar:

```
S -> > F V
F -> * V | f
V -> x | ( F V ) 
```
