"""Pnet is class of a parallel-series prefix network, based on networkx' MultiDiGraph
Used in induction algorithm and grammar analysis"""
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json
import math
import logging

from SFG.utils import hierarchy_pos, merge_nodes_and_keys, equivalence_partition

__all__ = ['Pnet']


class Pnet(nx.MultiDiGraph):
    """Pnet - a hybrid between Parallel-series network and prefix tree

    A parallel-series network can be defined inductively:

    * A network consisting of a single contact joining terminals is a parallel-series network
    * A network constructed from two parallel-series networks
      joined in parallel or in series is a parallel-series network.

    P-net is a parallel-series network with following properties:

    * Contains single node with no incoming edges (Start node)
    * Contains single node with no outgoing edges (End node)
    * Any paths between 2 nodes correspond to unique chain of symbols,
      made by concatination of edge labels of that path in order
    * For all nodes there can't be more than 1 outgoing edges marked with the same symbol

    Attributes
    ----------
    start : int
        Index of the Start node of the Pnet
    end : int
        Index of the End node of the Pnet

    See Also
    --------
    networkx : a network manipulation and analysis package

    """
    _end = -1
    _start = 0

    def __init__(self, data=None):
        """Initialise Pnet with a list of sentenses or another Pnet

        Note
        ----
        May also work if given list of list of hashable objects
        Though it was tested only on list of strings for now

        Parameters
        ----------
        data :  optional
            In case of Pnet given - copies it
            In case of MultiDiGraph - copies it and tries to find start and end nodes
            otherwise - initializes Pnet form the given container of sentences

        See Also
        --------
        add_sents
        """
        if isinstance(data, Pnet):
            super().__init__(data)
            self.start = data.start
            self.end = data.end
            self.next_node_id = data.next_node_id
            return

        if isinstance(data, nx.MultiDiGraph):
            super().__init__(data)
            self.start = next(node for node in data.nodes() if data.in_degree(node) == 0)
            self.end = next(node for node in data.nodes() if data.out_degree(node) == 0)
            self.next_node_id = 1 + max(data.nodes())
            return

        super().__init__()
        self.start = Pnet._start
        self.end = Pnet._end
        self.next_node_id = self.start + 1

        if data is not None:
            self.add_sents(data)

    @staticmethod
    def load(filename):
        """Load Pname from file

        Parameters
        ----------
        filename :  str

        See Also
        --------
        save
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            g = json_graph.node_link_graph(data)
            p = Pnet(g)

        return p

    def save(self, filename):
        """Save Pname to file

        Parameters
        ----------
        filename :  str

        See Also
        --------
        load
        """
        with open(filename, 'w') as f:
            data = json_graph.node_link_data(self)
            json.dump(data, f)

    def add_sents(self, sent_list, start=None, end=None, on_collision='warn'):
        """Add sentences to this Pnet

        Creates new chains between start and end,
        going through existing nodes whenever possible.

        This may cause collision if one sentence is a prefix of another one

        Characters in given sentences will be used as edge labels and keys

        Note
        ----
        May also work if given list of lists of hashable objects
        (this objects will be used as edge labels and keys)
        Though it was tested only on list of strings for now

        Parameters
        ----------
        sent_list : container of str
            list or any other iterable container of sentences to add
        start : int, default self.start
            node to start sentence from
        end : int, default self.end
            node where the sentence ends
        on_collision : {'warn', 'none', 'error'}
            what to do in case of collision

        See Also
        --------
        add_sent
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        for sent in sent_list:
            self.add_sent(sent, start, end, on_collision=on_collision)

    def add_sent(self, sent, start=None, end=None, on_collision='warn'):
        """Add a single sentence to this Pnet

        Creates new chain between start and end,
        going through existing nodes whenever possible

        This may cause collision if one sentence is a prefix of another one

        Note
        ----
        May also work if given list of hashable objects
        Though it was tested only on list of strings for now

        Parameters
        ----------
        sent_list : list of str
            list of sentences to add
        start : int, default self.start
            node to start sentence from
        end : int, default self.end
            node where the sentence ends
        on_collision : {'warn', 'none', 'error'}
            what to do in case of collision
        See Also
        --------
        add_sents
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        def handle_collision(sent):
            if on_collision == 'warn':
                logging.warning(f"Sentence '{sent}' caused a prefix collision and will be ignored")
            elif on_collision == 'none':
                pass
            else:
                raise ValueError(f"Sentence '{sent}' caused a prefix collision")

        prev = start

        for i, word_or_char in enumerate(sent):
            next_node = self.next_node_by_key(prev, word_or_char)
            if next_node == end:
                handle_collision(sent)
                return

            if next_node is None:
                self._add_sent(sent[i:], prev, end)
                return

            prev = next_node

        handle_collision(sent)

    def _add_sent(self, sent, start=None, end=None):
        prev = start if start is not None else self.start
        end = end if end is not None else self.end

        for i, word_or_char in enumerate(sent):
            # if we are on the final char - draw edge to the end node of the net
            if i == len(sent) - 1:
                self.add_edge(prev, end, label=word_or_char, key=word_or_char)
                break

            self.add_node(self.next_node_id)
            self.add_edge(prev, self.next_node_id, key=word_or_char)
            prev = self.next_node_id

            # just to be sure, may also help later in case of combined p-nets
            while self.next_node_id in self:
                self.next_node_id += 1

    def next_node_by_key(self, origin, key):
        """Get next node from origin node by key

        Parameters
        ----------
        origin : int
            origin node
        key : str
            edge key/label that leads to desired node from origin node

        Returns
        -------
        node : int or None
            returns desired node index

            OR

            None if no such node exists

        """
        if origin not in self:
            return None

        # edge[2] contains key
        labeled_edges = [edge for edge in self.out_edges(origin, keys=True) if edge[2] == key]

        if len(labeled_edges) == 0:
            return None

        if len(labeled_edges) > 1:
            raise ValueError("Net should not have outgoing edges starting with same symbol")

        return labeled_edges[0][1]

    def is_transit_node(self, node):
        """Check if given node has only 1 incoming and outgoing edge

        Parameters
        ----------
        node : int
            node to check

        Returns
        -------
        bool
        """
        return self.out_degree(node) == self.in_degree(node) == 1

    # def terminals(self):
    #     return set(nx.get_edge_attributes(self,'label').values())

    def length(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        return len(max(list(nx.all_simple_edge_paths(self, start, end)), key=len, default=[]))

    def sents(self, start=None, end=None, maxlen=None):
        """Generate sentences from paths between start and end nodes

        Parameters
        ----------
        start : int, default self.start
            node to start sentence from
        end : int, default self.end
            node where the sentence ends
        maxlen : int, default None
            maximum length (in the number of keys) of sentences to return,
            unlimited if equals to None

        Yields
        -------
        sent: list of str
            sentence as list of edges labels/keys

        See Also
        --------
        add_sents
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        paths = nx.all_simple_edge_paths(self, start, end, cutoff=maxlen)

        for path in paths:
            sent = []
            for _, _, word in path:
                sent += word
            yield sent

    def _remove_node_if_dead(self, node):
        if node not in self.nodes():
            return

        if self.out_degree(node) == 0 and node != self.end:
            self.remove_node_recursive(node)
            return

        if self.in_degree(node) == 0 and node != self.start:
            self.remove_node_recursive(node)
            return

    def remove_node_recursive(self, node):
        """Remove node and all resulting dangling nodes

        If after the node removal, any (except End and Start nodes)
        node will have 0 incoming or outgoing edges, it will also be removed

        Unlike ``Multigraph.remove_node``, ensures that Pnet stays valid_close

        Parameters
        ----------
        node : int
            node to remove

        See Also
        --------
        remove_edge_recursive
        """
        if node == self.start or node == self.end:
            return

        succ = list(self.successors(node))
        pred = list(self.predecessors(node))
        self.remove_node(node)

        for node in succ:
            self._remove_node_if_dead(node)

        for node in pred:
            self._remove_node_if_dead(node)

    def remove_edge_recursive(self, edge):
        """Remove edge and all resulting dangling nodes

        If after the edge removal, any (except End and Start nodes)
        node will have 0 incoming or outgoing edges, it will also be removed

        Unlike ``Multigraph.remove_node``, ensures that Pnet stays valid_close

        Parameters
        ----------
        edge : (int, int, str)
            edge to remove as a tuple of

            * start edge
            * end edge
            * key (edge label)

        See Also
        --------
        remove_node_recursive
        """
        s, e, _ = edge

        self.remove_edge(edge)

        self._remove_node_if_dead(s)
        self._remove_node_if_dead(e)

    def in_between(self, node, start=None, end=None, edge_cases=True):
        """Check if node is between two others

        Node *B* is between nodes *A* and *C*
        if there is path from *A* to *B*
        and from *B* to *C*

        Parameters
        ----------
        node : int
            node to check
        start : int, default self.start
            starting node
        end : int, default self.end
            end node
        edge_cases : bool, default True
            if False, start or end nodes are considered
            to be NOT in between each other

        Returns
        -------
        bool

        See Also
        --------
        between_nodes
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        if not edge_cases and (start == node or end == node):
            return False

        return nx.has_path(self, start, node) and nx.has_path(self, node, end)

    def between_nodes(self, start=None, end=None, edge_cases=True):
        """Get all nodes thar are between two given nodes

        Node *B* is between nodes *A* and *C*
        if there is path from *A* to *B*
        and from *B* to *C*

        Parameters
        ----------
        start : int, default self.start
            starting node
        end : int, default self.end
            end node
        edge_cases : bool, default True
            if False, start or end nodes are considered
            to be NOT in between each other

        Returns
        -------
        nodes : set of int
            nodes that are between ``start`` and ``end``

        See Also
        --------
        in_between
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        paths = nx.all_simple_paths(self, start, end)
        nodes = {node for path in paths for node in path}

        if not edge_cases:
            nodes.remove(start)
            nodes.remove(end)

        return nodes

    def similarity(self, other, self_start=None, self_end=None, other_start=None, other_end=None, t=None):
        """Check similarity of two Pnets or their parts

        If Pnets generate same sentences up to length ``t``,
        then they are considered similar

        Parameters
        ----------
        other : Pnet
            Pnet to compare this to

        self_start : int, default self.start
            start node in this Pnet for comparison
        self_end : int, default self.end
            end node in this Pnet for comparison

        other_start : int, default other.start
            start node in other Pnet for comparison
        other_end : int, default other.end
            end node in other Pnet for comparison

        t : int, default None
            maximum length (in the number of keys) of sentences to check,
            unlimited if equals to None

        Returns
        -------
        bool
        """
        self_start = self_start if self_start is not None else self.start
        self_end = self_end if self_end is not None else self.end

        other_start = other_start if other_start is not None else other.start
        other_end = other_end if other_end is not None else other.end

        s_sents = set(tuple(s) for s in self.sents(self_start, self_end, maxlen=t))

        # for every outgoing edge from start node, there must exist a path with a length <= t through this edge
        if not all(any(s[0] == k for s in s_sents) for _, _, k in self.out_edges(self_start, keys=True)):
            return False

        o_sents = set(tuple(s) for s in other.sents(other_start, other_end, maxlen=t))

        if not all(any(s[0] == k for s in o_sents) for _, _, k in other.out_edges(other_start, keys=True)):
            return False

        common = s_sents.intersection(o_sents)

        # o_sents is in s_sents OR s_sents is in o_sents
        if len(common) == len(o_sents) or len(common) == len(s_sents):
            return True

        return False

    def factorize(self, subnet, copy=True):
        """Attempt to factorize subnet

        Merge ancestors of given node (or subnet's end node),
        if they all have single path to a given node,
        and these paths correspond to the same sentence

        Parameters
        ----------
        subnet : (int, int)
            subnet to attempt factorization

        copy : bool, default True
            if False, modifies the object itself,
            instead of creating a new one

        Returns
        -------
        res : Pnet or None
            new Pnet if factorization was sucessful
            None if it cannot be done
        """

        s, e = subnet
        res = Pnet(self) if copy else self

        if res.in_degree(e) < 2:
            return None

        in_subnet_edges = [(prev, curr, k) for prev, curr, k in res.in_edges(e, keys=True) if nx.has_path(res, s, prev)]

        first_prev, _, key = in_subnet_edges[0]

        if all(k == key and res.is_transit_node(prev) for prev, _, k in in_subnet_edges):
            merge_nodes_and_keys(res, first_prev, [prev for prev, _, _ in in_subnet_edges])
            return res.factorize((s, first_prev), copy=False) or res

        return None

    def divide(self, subnet, subnet_tree=None, t=None, h=None):
        """Attempt to divide subnet

        Parameters
        ----------
        subnet : (int, int)
            subnet to attempt division
        subnet_tree : networkx.DiGraph, optional
            subnet hierarchy tree with additional labels
            if not given, it will be calculated automatically

        t : int, default None
            division parameter,
            used in similarity checks

        h : int, default None
            division parameter,
            used as a depth threshold

            if None - depth is unlimited

        Returns
        -------
        success : bool
            True if factorization was sucessful
            False if it cannot be done

        See Also
        --------
        similarity

        """
        h = h if h is not None else math.inf
        subnet_tree = subnet_tree if subnet_tree is not None else self.subnet_tree()
        s, e = subnet

        def division_equivalence(node1, node2):
            return self.similarity(self, node1, e, node2, e, t=t)

        deep_nodes = set()
        close_nodes = set()

        for node in self.between_nodes(s, e, edge_cases=False):
            if nx.shortest_path_length(self, s, node) > h:
                deep_nodes.add(node)
            elif self.out_degree(node) > 1:
                close_nodes.add(node)

        classes, partition = equivalence_partition(close_nodes, division_equivalence)

        # remove child nodes from classes
        # otherwise we may merge child with parent (extremely illegal)
        for eq_class in classes:
            for curr_node in list(eq_class):
                if any(curr_node != node and nx.has_path(self, node, curr_node) for node in eq_class):
                    eq_class.remove(curr_node)
                    partition.pop(curr_node)

        valid_close = []
        valid_deep = []
        paths = list(nx.all_simple_paths(self, s, e))
        for eq_class in classes:
            # ignore classes with less then 2 nodes to merge
            if len(eq_class) < 2:
                continue

            flag = False
            deep_part = set()
            for path in paths:
                flag = False
                for node in path:
                    if node in eq_class:
                        flag = True
                        break

                    if node in deep_nodes:
                        flag = True
                        deep_part.add(node)
                        break

                # no path through node in eq_class or through deep node
                if not flag:
                    break

            if flag:
                valid_close.append(eq_class)
                valid_deep.append(deep_part)

        if not valid_close:
            return None

        # print(close_nodes, deep_nodes, classes, partition, valid_close, valid_deep, '\n', sep='\n')

        # if many valid sets, select ones with smallest total amount of elements
        valids = zip(valid_close, valid_deep)
        valid_close, valid_deep = min(valids, key=lambda c_d: len(c_d[0]) + len(c_d[1]))

        first_node = valid_close.pop()
        net = Pnet(self)

        left_net = net.subcopy(s, first_node)
        right_net = net.subcopy(first_node, e)

        for node in valid_close:
            left_net = left_net.compose(net, other_start=s, other_end=node)
            right_net = right_net.compose(net, other_start=node, other_end=e)
            if right_net is None:
                return None

        new_subnet = Pnet.sequence_join(left_net, right_net)
        net.replace(new_subnet, s, e)

        return net

    def is_valid(self):
        """Checks if Pnets structure is valid

        Returns
        -------
        bool
        """
        tree = self.subnet_tree()
        for node in tree.nodes():
            for _, _, data in tree.out_edges(node, data=True):
                keys_to_child = set(data['keys'])
                if any(
                    keys_to_child.intersection(
                        data2['keys']) and data2['keys'] != data['keys'] for _,
                    _,
                    data2 in tree.out_edges(
                        node,
                        data=True)):
                    return False

        return True

    def _insert_inplace(self, other, target_node):
        """Here we split the node in two, and insert a net between them
            Nodes are splitted in such a way, that start and end nodes do not change
        """
        new_start = self.next_node_id
        self.add_node(new_start)
        self.next_node_id += 1

        if target_node == self.start:
            for s, e, k in list(self.out_edges(target_node, keys=True)):
                self.add_edge(new_start, e, key=k)
                self.remove_edge(s, e, key=k)

            target_node = new_start
            new_start = self.start
        else:
            # note that outgoing edges are stll go from target_node
            for s, e, k in list(self.in_edges(target_node, keys=True)):
                self.add_edge(s, new_start, key=k)
                self.remove_edge(s, e, key=k)

        return self.insert(other, new_start, target_node)

    def insert(self, other, start=None, end=None):
        """Insert a net between two nodes

        Modifies the existing Pnet

        Be sure to check if nets do not have the same keys at the start nodes
        If they do - insertion will rasie an error

        Parameters
        ----------
        other : Pnet
            other Pnet to insert
        start : int, default self.start
            start node in this Pnet for insertion
        end : int, default self.end
            end node in this Pnet for insertion

        Returns
        -------
        self : Pnet

        Raises
        ------
        ValueError
            If there is a key collision netween nets

        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        if start == end:
            return self._insert_inplace(other, start)

        self_keys = {k for _, _, k in self.out_edges(start, keys=True)}
        other_keys = {k for _, _, k in other.out_edges(other.start, keys=True)}

        if self_keys.intersection(other_keys):
            raise ValueError("Nets with same starting symbols were given")

        node_map = {n: n + self.next_node_id for n in other.nodes()}
        node_map[other.start] = start
        node_map[other.end] = end

        copy = nx.relabel_nodes(other, node_map)

        for s, e, k in copy.edges(keys=True):
            self.add_edge(s, e, k)

        self.next_node_id = 1 + max(node_map.values())

        return self

    def replace(self, other, start=None, end=None):
        """Replaces everything between two nodes with a given net

        There must be a path between start and end node

        Parameters
        ----------
        other : Pnet
            net to replace with
        start : int, default self.start
            starting node
        end : int, default self.end
            end node

        See Also
        --------
        between_nodes
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        dead_nodes = self.between_nodes(start, end, edge_cases=False)
        self.remove_nodes_from(dead_nodes)

        self.insert(other, start, end)

    def compose(self, other, self_start=None, self_end=None, other_start=None, other_end=None):
        """Compose two Pnets

        May fail due to collisions, in which case ``False`` is returned,
        and Pnet is left unchanged

        Note
        ----

        ``add_sents(other.sents())`` may be used as a faster composition alternative,
        but it does not keep the structure of other net

        Parameters
        ----------
        other : Pnet
            other Pnet to compose with

        self_start : int, default self.start
            start node in this Pnet for composition
        self_end : int, default self.end
            end node in this Pnet for composition

        other_start : int, default other.start
            start node in other Pnet for composition
        other_end : int, default other.end
            end node in other Pnet for composition

        Returns
        -------
        bool :
            new Pnet, if composition was successful

            OR

            None, if composition cannot be done
        """
        self_start = self_start if self_start is not None else self.start
        self_end = self_end if self_end is not None else self.end

        other_start = other_start if other_start is not None else other.start
        other_end = other_end if other_end is not None else other.end

        net = Pnet(self)

        if self_start == self_end:
            copy = other.subcopy(other_start, other_end)
            return net._insert_inplace(copy, self_start)

        other_to_self = {other_start: self_start, other_end: self_end}
        new_nodes = set()

        paths = nx.all_simple_edge_paths(other, other_start, other_end)

        for path in paths:
            for o_s, o_e, key in path:
                s_s = other_to_self[o_s]

                expected_s_e = other_to_self.get(o_e, None)
                actual_s_e = net.next_node_by_key(s_s, key)

                # (other) has node that do not exist in (self) yet
                if actual_s_e is None:
                    net.add_node(net.next_node_id)
                    actual_s_e = net.next_node_id
                    new_nodes.add(actual_s_e)
                    net.next_node_id += 1

                # not yet in the dict
                if expected_s_e is None:
                    expected_s_e = actual_s_e
                    other_to_self[o_e] = actual_s_e

                # 2 or more self nodes for 1 other node
                if expected_s_e != actual_s_e:
                    if expected_s_e in new_nodes:
                        merge_nodes_and_keys(net, actual_s_e, [expected_s_e])
                        other_to_self[o_e] = actual_s_e
                    elif actual_s_e in new_nodes:
                        merge_nodes_and_keys(net, expected_s_e, [actual_s_e])
                        other_to_self[o_e] = expected_s_e
                    else:
                        return None

                s_e = other_to_self[o_e]

                # 2 or more other nodes for 1 self node
                if len([other_node for other_node, self_node in other_to_self.items() if self_node == s_e]) > 1:
                    return None

                net.add_edge(s_s, s_e, key)

        # composition can break the parallel-series structure of Pnet
        # currently it is checked after and not during the composition
        # which is not very efficient
        if not net.is_valid():
            return None

        return net

    @staticmethod
    def sequence_join(start, *nets):
        res = Pnet(start)
        for net in nets:
            res._insert_inplace(net, res.end)

        return res

    @staticmethod
    def parallel_join(start, *nets):
        res = Pnet(start)

        for net in nets:
            res.insert(net)

        return res

    def subcopy(self, start=None, end=None):
        """Make a new Pnet from part of a current one

        Creates a new Pnet, that contains all nodes and edges
        between ``start`` and ``end`` nodes

        There needs to be a path between ``start`` and ``end``

        Parameters
        ----------
        start : int, default self.start
            node to start composition from,
            it will correspond to start node in the other Pnet
        end : int, default self.end
            node where the composition ends
            it will correspond to end node in the other Pnet

        Returns
        -------
        new_net : Pnet or None
            Resulting Pnet

            OR

            None if there is no path between ``start`` and ``end``,

        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        if not nx.has_path(self, start, end):
            return None

        subnodes = self.between_nodes(start, end)
        new_net = self.subgraph(subnodes).copy()

        new_net.start = start
        new_net.end = end
        new_net.next_node_id = 1 + max(new_net.nodes())
        return new_net

    def envelope_node(self, node, mode='start'):
        """Get the first subnet that contains ``node``

        Parameters
        ----------
        node : int
            node to envelope
        mode : {'start', 'end', 'inner'}, default 'start'
            if 'start' - return first subnet that starts from given node
            if 'end' - return first subnet that ends with given node
            if 'inner' - return first subnet that does not have node as its start or end

        Returns
        -------
        subnet : (int,int) or None
            The subnet as a tuple of its start and end nodes

            OR

            None if no such subnet can be found
        """
        if mode == 'start':
            if node == self.start:
                return (self.start, self.end)
            if node == self.end:
                return None

            out_edges = self.out_edges(node)
            for path_node in nx.shortest_path(self, node, self.end):
                if all(path_node == next or path_node in nx.descendants(self, next) for _, next in out_edges):
                    return (node, path_node)
            return None

        if mode == 'end':
            if node == self.end:
                return (self.start, self.end)
            if node == self.start:
                return None

            in_edges = self.in_edges(node)
            res = None
            for path_node in nx.shortest_path(self, self.start, node):
                if all(path_node == prev or path_node in nx.ancestors(self, prev) for prev, _ in in_edges):
                    res = (path_node, node)
            return res

        if mode == 'inner':
            if node == self.end or node == self.start:
                return None

            for path_node in nx.shortest_path(self, node, self.end):
                if path_node != node and any(node != prev and node not in nx.ancestors(self, prev)
                                             for prev, _ in self.in_edges(path_node)):
                    return self.envelope_node(path_node, mode='end')
            return (self.start, self.end)

    def envelope_subnet(self, subnet, subnets=None):
        """Get a first subnet that fully contains given ``subnet``

        Parameters
        ----------
        subnet : (int,int)
            subnet to envelope, as a tuple of its start and end nodes
        subnets : container of (int,int), optional
            list or other container with all non-trivial subnets of current Pnet
            if not given, it will be calculated automatically

        Returns
        -------
        subnet : (int,int) or None
            The subnet as a tuple of its start and end nodes

            OR

            None if no such subnet can be found

        See Also
        --------
        subnets
        """
        if subnet == (self.start, self.end):
            return None

        subnets = subnets if subnets is not None else self.subnets()
        subnet_lens = {(s, e): self.length(s, e) for s, e in subnets}

        s, e = subnet
        enveloping_subnets = [
            (this_s,
             this_e) for (
                this_s,
                this_e) in subnets if self.in_between(
                s,
                this_s,
                this_e) and self.in_between(
                e,
                this_s,
                this_e)]
        if (s, e) in enveloping_subnets:
            enveloping_subnets.remove((s, e))

        return min(enveloping_subnets, default=None, key=lambda subnet: subnet_lens[subnet])

    def subnets(self):
        """Get a set of all non-trivial subnets

        Subnet is a subgraph of Pnet, that is a valid_close Pnet itself,
        i.e it has Start and End nodes, and for all its inner nodes:
        * outgoing paths lead to End node
        * incoming paths go from Start node

        Non-trivial subnet is a subnet that is not just a chain of nodes

        Returns
        -------
        subnets : set of (int,int)
            The set of non-trivial subnets as a tuples of their start and end nodes
        """
        res = set()

        for node in self.nodes():
            if self.out_degree(node) > 1 or node == self.start:
                subnet = self.envelope_node(node, 'start')
                res.add(subnet)

            if self.in_degree(node) > 1 or node == self.end:
                subnet = self.envelope_node(node, 'end')
                res.add(subnet)

        return res

    def subnet_tree(self, subnets=None):
        """Get a hierarchy tree of non-trivial subnets

        In this tree root is the current Pnet itself
        And one subnet is the child of another if child subnet is fully contained in parent subnet


        Tree has an additional labels on nodes and egdes

        For nodes, label 'inner' contains tuple with keys of edges from starting node of the subnet,
        that are a part of this subnet
        (some start node's outgoing edges may be not a part of the subnet)

        For edges, label 'keys' contains tuple with keys of edges from start of the parent subnet,
        that lead to the start node of child subnet
        (or childs' 'inner' label, if child and parent start from the same node)

        Parameters
        ----------
        subnets : container of (int,int), optional
            list or other container with all non-trivial subnets of current Pnet
            if not given, it will be calculated automatically

        Returns
        -------
        tree : networkx.DiGraph
            Subnet hierarchy tree with additional labels

        See Also
        --------
        subnets, envelope_subnet
        """
        subnets = subnets if subnets is not None else self.subnets()

        tree = nx.DiGraph()

        for subnet in subnets:
            subnet_start, subnet_end = subnet
            envelope_subnet = self.envelope_subnet(subnet, subnets=subnets)

            inner_paths = nx.all_simple_edge_paths(self, subnet_start, subnet_end)
            inner_keys = tuple(set(k for (_, _, k), *_ in inner_paths))

            if envelope_subnet is None:
                tree.add_node(subnet, inner=inner_keys)
            else:
                envelop_start = envelope_subnet[0]

                if subnet_start == envelop_start:
                    paths = nx.all_simple_edge_paths(self, envelop_start, subnet_end)
                else:
                    paths = nx.all_simple_edge_paths(self, envelop_start, subnet_start)
                keys = tuple(set(k for (_, _, k), *_ in paths))
                tree.add_node(subnet, inner=inner_keys)
                tree.add_edge(envelope_subnet, subnet, keys=keys)

        return tree

    def draw_subnet_tree(self, subnet_tree=None, filename=None, dpi=192, show=True, **kwargs):
        """Draw a subnet hierarchy tree

        Drawing is done using matplotlib via networkx.draw

        Parameters
        ----------
        subnet_tree : networkx.DiGraph, optional
            subnet hierarchy tree with additional labels
            if not given, it will be calculated automatically
        filename : str, optional
            file name to save image to
            if None, image is not saved
        dpi : int, default 192
            saved image dpi, higher number mean bigger image
            (matplotlib does not allow to set image size)
        show : bool, default True
            if True, image will be shown
        **kwargs
            keyword arguments, that will be passed down to networkx.draw

        See Also
        --------
        subnet_tree, networkx.draw
        matplotlib : python data visualisation package
        """
        subnet_tree = subnet_tree if subnet_tree is not None else self.subnet_tree()
        pos = hierarchy_pos(subnet_tree, (self.start, self.end))

        fig = plt.figure()
        nx.draw(subnet_tree, ax=fig.add_subplot(111), pos=pos, with_labels=True, **kwargs)
        labels = nx.get_edge_attributes(subnet_tree, 'keys')
        nx.draw_networkx_edge_labels(subnet_tree, pos, edge_labels=labels)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

        if show:
            plt.show()

    def draw(
            self,
            scale_x=None,
            font_size=32,
            font_color='black',
            color=None,
            ec='black',
            cmap='summer',
            filename=None,
            dpi=192,
            show=True):
        """Draw a Pnet

        Drawing is done using matplotlib

        Parameters
        ----------
        scale_x : float, optional
            value the image (including font size) will be scaled along horizontal axis
            if not given, scale_x is calculated in a way that enusures maximum readability
        font_size : float, default 32
            base size of text on node and edge labels
        font_color : color, default 'black'
            color of text on node and edge labels,
            in any format supported by matplotlib
        cmap : str, default 'summer'
            node colormap name, among supported by matplotlib
        color : optional
            color in matplotlib's format
            'red, '#ffaacc', 0.4, (0.1,0,1) are all valid colors
            if specified, will be used instead of cmap for node coloring
        filename : str, optional
            file name to save image to
            if None, image is not saved
        dpi : int, default 192
            saved image dpi, higher number mean bigger image
            (matplotlib does not allow to set image size)
        show : bool, default True
            if True, image will be shown

        See Also
        --------
        matplotlib : python data visualisation package
        """
        fig, ax = plt.subplots(1)
        if color is None:
            cmap = plt.get_cmap(cmap)

        subnet_tree = self.subnet_tree()
        subnet_heights = {}

        def calc_height(subnet):
            # recursevly fills subnet_heights with heights
            inner_keys = set(subnet_tree.nodes[subnet]['inner'])

            if subnet_tree.out_degree(subnet) == 0:
                subnet_heights[subnet] = max(len(inner_keys), subnet_heights.get(subnet, 1))
                return subnet_heights[subnet]

            key_dict = {child_subnet: data['keys']
                        for _, child_subnet, data in subnet_tree.out_edges(subnet, data=True)}
            height_dict = {child_subnet: calc_height(child_subnet) for child_subnet in key_dict}

            key_height = {}
            keys_to_children = set()
            for child_subnet, keys in key_dict.items():
                keys_to_children.update(keys)
                key_height[keys] = max(height_dict[child_subnet], key_height.get(keys, 1))

            height = sum(key_height.values())
            for key in inner_keys:
                if key not in keys_to_children:
                    height += 1

            subnet_heights[subnet] = max(height, subnet_heights.get(subnet, 1))
            return height

        calc_height((self.start, self.end))
        origin = (0, 0)
        scale_x = scale_x if scale_x is not None else subnet_heights[(self.start, self.end)] / self.length()
        scale_y = 1
        base_arrow_offset = 0.5 * scale_y
        queue = [(self.start, origin)]
        completed = set()

        def node_visual_height(node):
            # node height is a max height of a subnet that starts or ends with this node
            filtered_subnets = {(s, e): h for (s, e), h in subnet_heights.items() if e == node or s == node}
            return 2 * max(filtered_subnets.values(), default=1) - 1

        def edge_visual_length(node, child):
            # get edge length as difference between node and child depth from origin
            child_depth = self.length(end=child)
            node_depth = self.length(end=node)
            return 2 * (child_depth - node_depth) - 1

        def label_center(ox, oy, dx, dy, text):
            centx = (ox + ox + dx) / 2
            centy = (oy + oy + dy) / 2
            plt.text(centx, centy, text, size=font_size * min(scale_x, 1 / scale_x),
                     va='center', ha='center', color=font_color)

        def edge_generator(node, subnet=None, arrow_offset=None):
            # iterates over edges in a subnet order
            # also calculates edge's length and offset
            arrow_offset = arrow_offset if arrow_offset is not None else base_arrow_offset
            if node == self.end:
                return

            if self.out_degree(node) == 1 and subnet is None:
                edge = s, e, k = next(iter(self.out_edges(node, keys=True)))
                edge_len = scale_x * edge_visual_length(s, e)
                yield (arrow_offset, edge_len, edge)
                return

            subnet = subnet if subnet is not None else self.envelope_node(node, mode='start')

            inner_keys = set(subnet_tree.nodes[subnet]['inner'])
            key_dict = {child_subnet: data['keys']
                        for _, child_subnet, data in subnet_tree.out_edges(subnet, data=True)}

            # if there is a chain of subnets, here will be the first ones of such chains
            first_subnets = {}
            keys_to_children = set()
            key_height = {}

            for child_subnet, keys in key_dict.items():
                key_height[keys] = max(subnet_heights[child_subnet], key_height.get(keys, 1))
                keys_to_children.update(keys)

                if keys not in first_subnets:
                    first_subnets[keys] = child_subnet
                else:
                    old_start, _ = first_subnets[keys]
                    _, new_end = child_subnet
                    if nx.has_path(self, new_end, old_start):
                        first_subnets[keys] = child_subnet

            for key in inner_keys:
                if key in keys_to_children:
                    continue

                end = self.next_node_by_key(node, key)
                edge_len = scale_x * edge_visual_length(node, end)
                edge = (node, end, key)
                yield (arrow_offset, edge_len, edge)
                arrow_offset += 2 * scale_y

            for keys, child_subnet in first_subnets.items():
                # if len > 1, than child_subnet starts from the same node as the current one
                # so we need to yield all of the child subnet's edges in one group
                if len(keys) > 1:
                    yield from edge_generator(node, child_subnet, arrow_offset)
                else:
                    key = keys[0]
                    end = self.next_node_by_key(node, key)
                    edge = (node, end, key)
                    edge_len = scale_x * edge_visual_length(node, end)
                    yield (arrow_offset, edge_len, edge)

                arrow_offset += 2 * key_height[keys]

        # main loop
        while queue:
            new_nodes = set()
            node, pos = queue.pop(0)

            if node in completed:
                continue

            height = scale_y * node_visual_height(node)
            if color is None:
                col = cmap(self.length(end=node) / self.length())
            else:
                col = color

            if ec is None:
                ecol = col
            else:
                ecol = ec
            rect = mpatches.Rectangle(pos, scale_x, height, color=col, ec=ecol)
            ax.add_patch(rect)
            label_center(pos[0], pos[1], scale_x, height, str(node))

            new_queue = []

            for edge_offset, edge_len, edge in edge_generator(node):
                _, child, key = edge

                apos = (pos[0] + scale_x, pos[1] + edge_offset)
                arrow = mpatches.Arrow(apos[0], apos[1], edge_len, 0, width=0.05 / scale_x, color='gray')
                ax.add_patch(arrow)
                label_center(apos[0], apos[1], edge_len, 0, str(key))

                if child not in new_nodes and child not in completed:
                    child_x = pos[0] + edge_len + scale_x
                    child_y = pos[1] + edge_offset - base_arrow_offset
                    new_nodes.add(child)
                    new_queue.append((child, (child_x, child_y)))

            completed.add(node)
            # put newest nodes first - thus making it kinda like depth-first search
            queue = new_queue + queue

        plt.axis('equal')
        plt.axis('off')

        # removes white padding in the saved image
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

        if show:
            plt.show()
