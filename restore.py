import networkx as nx
from nltk.parse.recursivedescent import RecursiveDescentParser
from nltk import CFG
from itertools import product

from pnet import Pnet
from utils import equivalence_partition


def restore(sents):
    """Create grammars that can produce given sentences

    Grammar syntax example:

    S -> 'c' A 'a' B | 'b'

    A -> 'a' A | 'A'

    B -> 'b' A

    Parameters
    ----------
    sents: collection of str
        sentences to use in restoration

    Returns
    -------
    grammars : set of str
        grammars that can produce given sentences
    """
    maxlen = len(max(sents, key=len))
    res = set()
    for t,h in product(range(1, maxlen+1), repeat=2):
        p = Pnet(sents)
        net_transform(p, t, h)
        _, g = net_to_grammar(p, t)

        if all(check_grammar(g, s) for s in sents):
            # print(f'Success with t={t}, h={h}')
            # print(g, '\n')
            res.add(g)

    return res

def check_grammar(grammar, sent):
    """Check if sentence can be produced by grammar

    Grammar syntax example:

    S -> 'c' A 'a' B | 'b'

    A -> 'a' A | 'A'

    B -> 'b' A

    Parameters
    ----------
    grammar : str
        grammar to check

    sents: str
        sentence to check

    Returns
    -------
    bool
    """
    g = CFG.fromstring(grammar)
    parser = RecursiveDescentParser(g)

    try:
        if not parser.parse(sent):
            return False

    except ValueError:
        return False

    return True

def _net_transform_step(net, step_type='factorization', t=None, h=None):
    tree = net.subnet_tree()
    root = (net.start, net.end)
    
    i = 0
    queue = [root]

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
            # if success:
            #   # print(f'Success {step_type} of {subnet}')

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
        # net.draw(font_size=12, filename=f'net_transform-{i}.png', show=False)
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

    non_terms = {net: str(classes.index(eclass)) for net, eclass in partition.items()}
    root = (net.start, net.end)
    queue = [root]

    S = non_terms[root]

    completed = set()
    rules = []

    while queue:
        curr = queue.pop()
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