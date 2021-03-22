from nltk import CFG, Nonterminal, Production
from nltk.grammar import is_nonterminal, is_terminal
import itertools
import sys


def terminals(g):
    """Get set of grammar's terminals

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    terminals : set of hashable

    See Also
    --------
    nltk.CFG
    """
    return set(g._lexical_index.keys())

def nonterminals(g):
    """Get set of grammar's nonterminals

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    terminals : set of nltk.Nonterminal

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    return g._categories.copy()

def children(g, parent):
    """Get Nonterminals that are used in a production or nonterminal productions   

    Parameters
    ----------
    g : nltk.CFG

    parent : nltk.Production or nltk.Nonterminal

    Returns
    -------
    children : set of Nonterminal

    See Also
    --------
    nltk.CFG, nltk.Nonterminal, nltk.Production
    """
    res = set()

    if isinstance(parent, Production):
        prods = [parent]
    else:
        prods = g.productions(parent)

    for prod in prods:
        for item in prod.rhs():
                if is_nonterminal(item):
                    res.add(item)

    return res

def endings(g, n):
    """Get right hand sides that consist only of terminals

    Parameters
    ----------
    g : nltk.CFG

    n : nltk.Nonterminal

    Returns
    -------
    children : set of Nonterminal

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    res = set()

    for prod in g.productions(n):
        if all(not is_nonterminal(item) for item in prod.rhs()):
            res.add(prod.rhs())

def remove_production(g, prod):
    """Remove production from a grammar

    Parameters
    ----------
    g : nltk.CFG

    prod : nltk.Production

    Returns
    -------
    nltk.CFG

    See Also
    --------
    nltk.CFG, nltk.Production
    """
    if len(g.productions(prod.lhs())) == 1:
        return remove_nonterminal(g, prod.lhs())

    prods = [p for p in g.productions() if p != prod]

    return CFG(prods)

def remove_nonterminal(g, nont):
    """Remove nonterminal from a grammar

    Parameters
    ----------
    g : nltk.CFG

    nont : nltk.Nonterminal

    Returns
    -------
    nltk.CFG

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    prods = [p for p in g.productions() if p.lhs() != nont and nont not in p.rhs()]

    return CFG(prods)

def add_production(g, prod):
    """Add production to a grammar

    Parameters
    ----------
    g : nltk.CFG

    prod : nltk.Production

    Returns
    -------
    nltk.CFG

    See Also
    --------
    nltk.CFG, nltk.Production
    """
    prods = list(g.productions())
    prods.append(prod)

    return CFG(prods)

def is_separated(g):
    """Check if grmmar is separated

    Grammar is separated if all its productions start with a terminal

    And for all nonterminals, no two productions start with the same terminal

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    bool

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    nonts = nonterminals(g)

    for nont in nonts:
        starts = set()
        for prod in g.productions(nont):
            start = prod.rhs()[0]

            if is_nonterminal(start):
                return False

            if start in starts:
                return False

            starts.add(start)
 
    return True

def unreachable(g):
    """Get set of unreachable nonterminals in grammar

    Example of grammar with unreachable nonterminal:

    S -> 'a' A | 'b'

    A -> 'a' A | '0'

    B -> 'b' A

    Here B is unreachable

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    children : set of Nonterminal

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    queue = {g.start()}
    completed = set()

    while queue:
        n = queue.pop()
        completed.add(n)

        for child in children(g,n):         
            if child not in completed:
                queue.add(child)

    nonts = set(nonterminals(g))

    return nonts.difference(completed)

def useless(g):
    """Get set of useless nonterminals in grammar

    Example of grammar with useless nonterminal:

    S -> 'a' A | 'b' B

    A -> 'a' A | '0'

    B -> 'b' B

    Here B is useless

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    children : set of Nonterminal

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    nonts = set(nonterminals(g))

    useful = {n for n in nonts if endings(g,n)}
    change = True

    while change:
        change = False

        for n in nonts.difference(useful):
            for prod in g.productions(n):
                if all(child in useful for child in children(g, prod)):
                    useful.add(n)
                    change = True
                    break

    return nonts.difference(useful) 

def nonterm_equal(g, n1, n2):
    """Check if nonterminals may produce the same language

    Assumes that other nonterminals all produce different languages 

    Parameters
    ----------
    g : nltk.CFG

    n1 : nltk.Nonterminal

    n2 : nltk.Nonterminal

    Returns
    -------
    bool

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    p1 = g.productions(n1)
    p2 = g.productions(n2)

    if len(p1) != len(p2):
        return False

    rules1 = {p.rhs() for p in p1}
    rules2 = {p.rhs() for p in p2}

    temp = rules1.copy()

    rules1.difference_update(rules2)
    rules2.difference_update(temp)

    if not rules1:
        return True

    def rename(t, from_, to):
        if from_ not in t:
            return tuple(t)

        l = list(t)
        i = l.index(from_)
        l[i] = to

        return rename(l, from_, to)

    rules1 = {rename(r, n2, n1) for r in rules1}
    rules2 = {rename(r, n2, n1) for r in rules2}

    if rules1.difference(rules2):
        return False
    else:
        return True

def check_canonical(g):
    """Check grammar for canonical rules violation

    These rules are (in simple words):

    #. Starting symbol must not appear in any rhs
    #. There should be no useless or unreachable nonterminals
    #. All nonterminals, except starting one, must have more than one production
    #. For all nonterminals, except starting one, all its productions must not end with the same terminal
    #. Every pair of nonterminals, except with starting one, must produce different languages
    #. For all nonterminals, except starting one, all its productions must not end with the same nonterminal

    Parameters
    ----------
    g : nltk.CFG
        Must be also separated grammar

    Returns
    -------
    broken_rules : set of int:
        Set with broken rule numbers that have been broken
        So, if this set is empty - grammar may be canonical

    See Also
    --------
    nltk.CFG, nltk.Nonterminal
    """
    if not is_separated(g):
        raise ValueError("Non-separated grammar was given")

    nonts = nonterminals(g)

    broken_rules = set()

    ends = {nont: set() for nont in nonts}
    counts = {nont: 0 for nont in nonts}

    for prod in g.productions():
        ends[prod.lhs()].add(prod.rhs()[-1])
        counts[prod.lhs()] += 1

        for item in prod.rhs():
            if item == g.start():
                broken_rules.add(1)

    for end in ends.values():
        if len(end) == 1:
            if is_nonterminal(end.pop()):
                broken_rules.add(6)
            else:
                broken_rules.add(4)

    for nont, num in counts.items():
        if nont == g.start():
            continue

        if num == 1:
            broken_rules.add(3)

    trash1 = useless(g)
    trash2 = unreachable(g)

    if trash1 or trash2:
        broken_rules.add(2)

    for n1,n2 in itertools.combinations(nonts, 2):
        if nonterm_equal(g, n1, n2):
            broken_rules.add(5)
 
    return broken_rules

def apply_production(sent, prod):
    res = []
    for item in sent:
        if is_nonterminal(item) and item == prod.lhs():
            res.extend(prod.rhs())
        else:
            res.append(item)

    return res


def generate(grammar, start=None, depth=None, maxlen=None, n=None):
    """Generates an iterator of all sentences from a CFG.

    Parameters
    ----------
    grammar : nltk.CFG
        The Grammar used to generate sentences
    start : nltk.Nonterminal
        The Nonterminal from which to start generate sentences
    depth : int
        The maximal depth of the generated tree
    maxlen : int
        The maximal length of generated sentences
    n : int
        The maximum number of sentences to return

    Yields
    -------
    iter :
        An iterator of lists of terminal tokens.
    """
    if not start:
        start = [grammar.start()]
    if depth is None:
        depth = sys.maxsize
    if maxlen is None:
        maxlen = sys.maxsize

    iter = _generate_all(grammar, start, depth, maxlen)

    if n:
        iter = itertools.islice(iter, n)

    return iter



def _generate_all(grammar, items, depth, maxlen):
    if items:
        for frag1 in _generate_one(grammar, items[0], depth, maxlen):
            for frag2 in _generate_all(grammar, items[1:], depth, maxlen - len(frag1)):
                yield frag1 + frag2

    else:
        yield []


def _generate_one(grammar, item, depth, maxlen):
    if depth > 0 and maxlen > 0:
        if is_nonterminal(item):
            for prod in grammar.productions(lhs=item):
                for frag in _generate_all(grammar, prod.rhs(), depth - 1, maxlen):
                    yield frag
        else:
            yield [item]

    if depth > 0 and maxlen == 0 and is_nonterminal(item):
        # has empty production
        if any(prod.rhs() == tuple() for prod in grammar.productions(item)):
            yield []
