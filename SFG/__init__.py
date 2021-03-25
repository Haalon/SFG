"""
SFG
===

Grammar inference is a process of producing a formal grammar from a set of given sentences

Context-Free grammar is separated if all its productions start with a terminal
And for all nonterminals, no two productions start with the same terminal


Available modules
-----------------
grammar
    Additional, useful functions that work on context-free grammars from `nltk`
pnet
    Pnet is class of a parallel-series prefix network, based on networkx' MultiDiGraph
    Used in induction algorithm and grammar analysis
restore
    Functions, related to grammar induction

"""
import SFG.grammar
import SFG.restore
import SFG.pnet

# __all__ = SFG.grammar.__all__ + SFG.restore.__all__ + SFG.pnet.__all__