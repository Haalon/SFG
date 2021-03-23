import networkx as nx
from nltk.parse.recursivedescent import RecursiveDescentParser
from nltk import CFG
import itertools
from functools import cmp_to_key
from pnet import Pnet
from utils import equivalence_partition
from grammar import *

def _cmp_grammar_simplicity(g1, g2):
    if g1.count('\n') < g2.count('\n'):
        return -1
    elif g1.count('\n') > g2.count('\n'):
        return 1

    if g1.count('|') < g2.count('|'):
        return -1
    elif g1.count('|') > g2.count('|'):
        return 1

    if g1.count("'") < g2.count("'"):
        return 1
    elif g1.count("'") > g2.count("'"):
        return -1

    return 0


def restore(sents, maxt=None, maxh=None):
    """Create grammars that can produce given sentences

    Grammar syntax example:

    S -> 'c' A 'a' B | 'b'

    A -> 'a' A | 'A'

    B -> 'b' A

    Parameters
    ----------
    sents: collection of str
        sentences to use in restoration

    maxt: int
        check up values of t up to this parameter

    maxh: int
        check up values of h up to this parameter

    Returns
    -------
    grammars : dict of str
        grammar strings for every valid pair of t and h
    """
    maxlen = len(max(sents, key=len))
    maxt = maxt if maxt is not None else maxlen
    maxh = maxh if maxh is not None else maxlen

    res = {}
    for t,h in itertools.product(range(1, maxt+1), range(1, maxh+1)):
        p = Pnet(sents)
        net_transform(p, t, h)
        _, g = net_to_grammar(p, t)

        g = CFG.fromstring(g)

        if all(check_grammar(g, s) for s in sents):
            print(f'Success with t={t}, h={h}')
            print(g, '\n')
            res[(t,h)] = g
        else:
            print(f'Fail with t={t}, h={h}')

    return res

def check_grammar(g, sent):
    """Check if sentence can be produced by grammar

    Grammar syntax example:

    S -> 'c' A 'a' B | 'b'

    A -> 'a' A | 'A'

    B -> 'b' A

    Parameters
    ----------
    g : nltk.CFG
        grammar to check

    sents: str
        sentence to check

    Returns
    -------
    bool

    See Also
    --------
    CFG.grammar
    """
    parser = RecursiveDescentParser(g)

    try:
        if not list(parser.parse(sent)):
            return False

    except ValueError:
        return False

    return True

def _net_transform_step(net, step_type='factorization', t=None, h=None):
    tree = net.subnet_tree()
    root = (net.start, net.end)
    
    i = 0
    queue = [root]

    # print(f'\n{step_type} attempt with t={t} h={h}')
    while queue:
        flag = False
        for subnet in queue:
            if step_type == 'factorization':
                success = net.factorize(subnet)
            elif step_type == 'division':
                success = net.divide(subnet, tree, t, h)
            else:
                raise ValueError

            flag = flag or success
            if success:
                print(f'\tSuccess {step_type} of {subnet} ({i})')
            # else:
            #     print(f'\tFailed {step_type} of {subnet} ({i})')

        if flag:
            return True
        i += 1

        queue = nx.descendants_at_distance(tree, root, i)

    return False


def net_transform(net, t=None, h=None):
    """Simplify Pnet for grammar restoration

    Parameters
    ----------
    net : Pnet
        net to transform

    t : int, default None
        division parameter,
        used in similarity checks

    h : int, default None
        division parameter,
        used as a depth threshold

        if None - depth is unlimited

    See Also
    --------
    pnet.Pnet, pnet.Pnet.divide, restore
    """
    flag = True
    i = 0
    while flag:
        # net.draw(font_size = 50, filename=f'algo{i}.png')
        i+=1
        flag = _net_transform_step(net, 'factorization',t,h)
        if flag:
            continue
        flag = _net_transform_step(net, 'division',t,h)


def _subnet_to_rule(net, subnet, subnet_tree, non_terms):
    sub_children = list(subnet_tree.successors(subnet))
    
    s_to_e = {}
    for s,e in sub_children:
        if s not in s_to_e:
            s_to_e[s] = [e]
        else:
            s_to_e[s].append(e)


    paths = nx.all_simple_edge_paths(net, subnet[0], subnet[1])
    res = set()

    for path in paths:
        subnet_s = None
        variant = ''
        for s,e,k in path:
            if subnet_s is not None and s in s_to_e[subnet_s]:
                variant += ' ' + non_terms[(subnet_s, s)]
                subnet_s = None

            if s in s_to_e:
                subnet_s = s

            if subnet_s is not None:
                continue

            variant +=  ' ' + f"'{k}'"

        # if child subnet ends on the same node as parent
        if subnet_s is not None:
            variant += ' ' + non_terms[(subnet_s, subnet[1])]

        res.add(variant)

    rule = non_terms[subnet] + ' ->' + ' |'.join(res)
    return rule

def net_to_grammar(net, t=None):
    """Restore a grammar that corresponds to a given Pnet

    Parameters
    ----------
    net : Pnet
        net to transform

    t : int, default None
        division parameter,
        used in similarity checks

    Returns
    -------
    (start, grammar) :
        starting symbol and a grammar in a string format

    See Also
    --------
    pnet.Pnet, restore
    """   
    def subnet_eqivalence(net1, net2):
        s1, e1 = net1
        s2, e2 = net2
        return net.similarity(net,s1,e1,s2,e2,t=t)

    subnet_tree = net.subnet_tree()
    classes, partition = equivalence_partition(subnet_tree.nodes, subnet_eqivalence)
    root = (net.start, net.end)

    def subnet_order(net1, net2):
        if nx.has_path(subnet_tree, net1, net2):
            return -1

        if nx.has_path(subnet_tree, net2, net1):
            return 1

        minkey1 = min(next(iter(subnet_tree.in_edges(net1, data=True)))[2]['keys'])
        minkey2 = min(next(iter(subnet_tree.in_edges(net1, data=True)))[2]['keys'])

        if minkey1 < minkey2:
            return -1
        elif minkey1 > minkey2:
            return 1

        if nx.has_path(net, net1[1], net2[0]):
            return -1
        else:
            return 1

    # nets are ordered so that Pnets with the same structure but different edge order 
    # will produce grammars with same order of rules and alternatives
    ordered = sorted(subnet_tree.nodes, key=cmp_to_key(subnet_order))
    non_terms = {net: str(classes.index(partition[net])) for net in ordered}
    
    queue = [root]

    S = non_terms[root]

    completed = set()
    rules = []

    while queue:
        curr = queue.pop(0)

        if non_terms[curr] in completed:
            continue

        rule = _subnet_to_rule(net, curr, subnet_tree, non_terms)
        rules.append(rule)
        completed.add(non_terms[curr])

        for child in subnet_tree.successors(curr):
            nonterm = non_terms[child]
            if nonterm not in completed:
                queue.append(child)

    return S, '\n'.join(rules)

def _generate_sents(g, target, maxlen):
    res = set()
    for prod in g.productions(target):
        sents = list(generate(g, prod.rhs(), maxlen=maxlen))

        if not sents:
            return set()

        res.update(tuple(s) for s in sents)

    return res

def _minimal_different_sents(g):
    def differs(sset, list):
        count = 0
        for s in list:
            if not sset.difference(s):
                count+=1

            if count > 1:
                return False

        return True

    queue = nonterminals(g)
    res = {}

    maxlen = 1
    while queue:
        sets = {nont: _generate_sents(g, nont, maxlen) for nont in queue}

        for nont, sents in sets.items():
            if sents and differs(sents, sets.values()):
                res[nont] = sents
                queue.remove(nont)

        maxlen += 1

    return res

def min_pnet(g):
    """Generate a minimal Pnet that can be restored back to grammar

    Also calculates `t` and `h` values for a grammar
    These values are properties of the grammar, and used in restoration
    
    They are accessible via `graph` field of the Pnet

    Parameters
    ----------
    g : nltk.CFG

    Returns
    -------
    net : Pnet

    See Also
    --------
    pnet.Pnet, nltk.grammar
    """   
    nont_sents = _minimal_different_sents(g)

    t = len(max((s for sents in nont_sents.values() for s in sents), key = len))
    nets = {nont: Pnet(sents) for nont,sents in nont_sents.items()}
    start = g.start()

    res = Pnet(prod.rhs() for prod in g.productions(start))
    res.graph['t'] = t

    completed = {start}
    change = True

    while change:
        change = False
        # res.draw()

        for (s,e,k) in list(res.edges(keys=True)):
            if is_nonterminal(k):
                if k in completed:
                    res.remove_edge(s,e,k)
                    res.compose(nets[k], self_start=s, self_end=e)
                else:
                    change = True
                    completed.add(k)
                    temp = Pnet(prod.rhs() for prod in g.productions(k))
                    res.remove_edge(s,e,k)
                    res.compose(temp, self_start=s, self_end=e)

    tree = res.subnet_tree()

    h = 0
    for subnet in tree.nodes():
        parent_start = subnet[0]
        for child_net in tree.successors(subnet):
            child_start = child_net[0]

            pathlen = len(max(nx.all_simple_edge_paths(res, parent_start,  child_start), key=len))
            h = max(h, pathlen)

    res.graph['h'] = h


    return res