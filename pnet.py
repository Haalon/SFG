import networkx as nx
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

from utils import hierarchy_pos

class Pnet(nx.MultiDiGraph):
    """Pnet - a hybrid between Parallel-series network and prefix tree

    A parallel-series network can be defined inductively: 
    * A network consisting of a single contact joining terminals is a parallel-series network
    * A network constructed from two parallel-series networks joined in parallel or in series is a parallel-series network.

    P-net is a parallel-series network with following properties:

    * Contains single node with no incoming edges (Start node)
    * Contains single node with no outgoing edges (End node)
    * Any paths between 2 nodes correspond to unique chain of symbols,
      made by concatination of edge labels of that path in order
    * For all nodes there can't be more than 1 outgoing edges marked with the same symbol

    This class is built around networkx' MultiDiGraph class

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
        data : Pnet or container  of str, optional
            In case of Pnet given - copies it
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

        super().__init__()
        self.start = Pnet._start
        self.end = Pnet._end
        self.next_node_id = self.start + 1

        if data is not None:
            self.add_sents(data)

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
            if   on_collision == 'warn':
                logging.warning(f"No sentence should be a prefix of another one. Sentence '{sent}' will be ignored")
            elif on_collision == 'none':
                pass
            else:
                raise ValueError(f"No sentence should be a prefix of another one. Sentence '{sent}' caused a collision")

        prev = start

        for i, word_or_char in enumerate(sent):
            next_node = self.next_node_by_key(prev, word_or_char)
            if next_node == end:
                handle_collision(sent)
                return

            if next_node == None:
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
                break;

            
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
            returns desired node index OR None if no such node exists

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

    def sents(self, start=None, end=None, cutoff=None):
        """Generate sentences from paths between start and end nodes

        Parameters
        ----------
        start : int, default self.start
            node to start sentence from
        end : int, default self.end
            node where the sentence ends
        cutoff : int, default None
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

        paths = nx.all_simple_edge_paths(self, start, end, cutoff=cutoff)
        
        for path in paths:
            sent = []
            for _,_, word in path:
                sent += word
            yield sent

    def similarity(self, other, t=None):
        """Check similarity of two Pnets

        If Pnets generate same sentences up to length ``t``,
        then they are considered similar

        Parameters
        ----------
        other : Pnet
            Pnet to compare this to
        t : int, default None
            maximum length (in the number of keys) of sentences to check,
            unlimited if equals to None

        Returns
        -------
        bool
        """
        o_sents = set(tuple(s) for s in other.sents(cutoff=t))
        s_sents = set(tuple(s) for s in self.sents(cutoff=t))

        return False if o_sents.symmetric_difference(s_sents) else True

    def _merge_nodes(self, keep_node, merge_nodes):
            nx.relabel_nodes(self, {m_node: keep_node for m_node in merge_nodes}, copy=False)
            # relabeling in netwrokx may create new unwanted edges automatically
            for (s,e,k) in self.out_edges(keep_node, keys=True):
                if isinstance(k, int):
                    self.remove_edge(s,e,k)
            for (s,e,k) in self.in_edges(keep_node, keys=True):
                if isinstance(k, int):
                    self.remove_edge(s,e,k)

    def factorize(self, subnet_or_node):
        """Right factorize subnet or node

        Merge ancestors of given node (or subnet's end node),
        if they all have single path to a given node,
        and this paths correspond to the same sentence

        Parameters
        ----------
        subnet_or_node : (int, int) or int
            maximum length (in the number of keys) of sentences to check,
            unlimited if equals to None

        Returns
        -------
        success : bool
            True if factorization was sucessful
            False if it cannot be done
        """
        if isinstance(subnet_or_node, int):
            e = subnet_or_node
        else:
            _,e = subnet_or_node

        if self.in_degree(e) < 2:
            return False

        prev,_,key = next(self.in_edges(e, keys=True))

        if all(k==key and self.is_transit_node(s) for s,_,k in self.in_edges(e, keys=True)):
            self._merge_nodes(prev, self.predecessors(e))

        return self.factorize(prev) or True

    def compose(self, other, start=None, end=None):
        """Compose two Pnets

        Add edges and nodes from other Pnet, that do not exist in a current one,
        while preserving ones that exist in current or in both Pnets

        May fail due to collisions, in which case current Pnet is left unchanged

        Parameters
        ----------
        other : Pnet
            other Pnet to compose with
        start : int, default self.start
            node to start composition from,
            it will correspond to start node in the other Pnet
        end : int, default self.end
            node where the composition ends
            it will correspond to end node in the other Pnet

        Returns
        -------
        success : bool
            True if compose was sucessful
            False if it cannot be done
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        backup = Pnet(self)

        other_to_self = {other.start: start, other.end: end}
        new_nodes = set()

        paths = nx.all_simple_edge_paths(other, other.start, other.end)

        for path in paths:
            for o_s, o_e, key in path:
                s_s = other_to_self[o_s]

                expected_s_e = other_to_self.get(o_e, None)
                actual_s_e = self.next_node_by_key(s_s, key)
                
                # (other) has node that do not exist in (self) yet
                if actual_s_e is None:
                    self.add_node(self.next_node_id)
                    actual_s_e = self.next_node_id
                    new_nodes.add(actual_s_e)
                    self.next_node_id += 1

                # not yet in the dict
                if expected_s_e is None:
                    expected_s_e = actual_s_e
                    other_to_self[o_e] = actual_s_e

                # 2 or more self nodes for 1 other node
                if expected_s_e != actual_s_e:
                    if expected_s_e in new_nodes:
                        self._merge_nodes(actual_s_e, [expected_s_e])
                        other_to_self[o_e] = actual_s_e
                    elif actual_s_e in new_nodes:
                        self._merge_nodes(expected_s_e, [actual_s_e])
                        other_to_self[o_e] = expected_s_e
                    else:
                        self = backup
                        return False

                s_e = other_to_self[o_e]

                # 2 or more other nodes for 1 self node
                if len([other_node for other_node,self_node in other_to_self.items() if self_node==s_e]) > 1:
                    self = backup
                    return False

                self.add_edge(s_s, s_e, key)

        return True

    def collapse(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        dead_nodes = [node for node in self if self.in_subnet((start, end), node, edge_cases=False)]

        for s,e,k in list(self.in_edges(start, keys=True)):
            self.remove_edge(s,e,k)
            self.add_edge(s, end, k)

        self.remove_nodes_from(dead_nodes)
        self.remove_node(start)

    def cut(self, start=None, end=None):
        """Make a new Pnet from part of a current one

        Creates a new Pnet, that contains all nodes and edges
        between ``start`` and ``end`` nodes in a current one

        There needs to be a path ``start`` and ``end``

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
            None if there is no path between ``start`` and ``end``,
            Resulting Pnet otherwise
        """
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        desc = nx.descendants(self, start)
        desc.add(start)

        if end not in desc:
            return None

        ancs = nx.ancestors(self, end)
        ancs.add(end)
        new_net = self.subgraph(desc.intersection(ancs)).copy()

        nx.relabel_nodes(new_net, {end: Pnet._end}, copy=False)
        new_net.start = start
        new_net.end = Pnet._end
        new_net.next_node_id = end
        return new_net

    def in_subnet(self, subnet, node, edge_cases=True):
        """Check if subnet contains node

        Parameters
        ----------
        subnet : (int, int)
            subnet to check
        node : int
            node to check
        edge_cases : bool, default True
            if True, start or end node of subnet
            is considered to be inside this subnet

        Returns
        -------
        bool
        """
        s_start, s_end = subnet

        if edge_cases and (s_start == node or s_end == node):
            return True

        desc = nx.descendants(self, s_start)
        ancs = nx.ancestors(self, s_end)

        return node in desc and node in ancs

    def envelope_node(self, node, mode='start'):
        """Get the first subnet that contains node

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

            OR None if no such subnet can be found
        """
        if mode == 'start':
            if node == self.start:
                return (self.start, self.end)
            if node == self.end:
                return None

            out_edges = self.out_edges(node)
            for path_node in nx.shortest_path(self, node, self.end):
                if all(path_node == next or path_node in nx.descendants(self, next) for _,next in out_edges):
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
                if all(path_node == prev or path_node in nx.ancestors(self, prev) for prev,_ in in_edges):
                    res = (path_node, node)
            return res

        if mode == 'inner':
            if node == self.end or node == self.start:
                return None
            
            for path_node in nx.shortest_path(self, node, self.end):
                if path_node != node and any(node != prev and node not in nx.ancestors(self, prev) for prev,_ in self.in_edges(path_node)):
                    return self.envelope_node(path_node, mode='end')
            return (self.start, self.end)

    def envelope_subnet(self, subnet, subnets=None):
        """Get a first subnet that fully contains given subnet

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

            OR None if no such subnet can be found

        See Also
        --------
        subnets
        """
        if subnet == (self.start, self.end):
            return None

        subnets = subnets if subnets is not None else self.subnets()
        subnet_lens = {(s,e): self.length(s,e) for s,e in subnets}        
        
        s, e = subnet
        enveloping_subnets = [subnet for subnet in subnets if self.in_subnet(subnet, s) and self.in_subnet(subnet, e)]
        enveloping_subnets.remove((s, e))

        return min(enveloping_subnets, default=None, key=lambda subnet: subnet_lens[subnet])

    def subnets(self):
        """Get a set of all non-trivial subnets

        Subnet is a subgraph of Pnet, that is a valid Pnet itself,
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

        For nodes, label 'inner' contains tuple with keys of edges from start of the subnet,
        that are a part of this subnet
        (some start node's outgoing edges may be not a part of this subnet)

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
            inner_keys = tuple(set(k for (_,_,k), *_ in inner_paths))
            
            if envelope_subnet is None:
                tree.add_node(subnet, inner=inner_keys)
            else:                
                envelop_start = envelope_subnet[0]
                
                if subnet_start == envelop_start:
                    paths = nx.all_simple_edge_paths(self, envelop_start, subnet_end)
                else:
                    paths = nx.all_simple_edge_paths(self, envelop_start, subnet_start)
                keys = tuple(set(k for (_,_,k), *_ in paths))                
                tree.add_node(subnet, inner=inner_keys)
                tree.add_edge(envelope_subnet, subnet, keys=keys)

        return tree

    def draw_subnet_tree(self, subnet_tree=None, filename=None, dpi=960, show=True, **kwargs):
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
        dpi : int, default 960
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
        pos = hierarchy_pos(subnet_tree,(self.start, self.end))

        fig = plt.figure()
        nx.draw(subnet_tree, ax=fig.add_subplot(111), pos=pos, with_labels=True, **kwargs)
        labels = nx.get_edge_attributes(subnet_tree,'keys')
        nx.draw_networkx_edge_labels(subnet_tree,pos, edge_labels=labels)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

        if show:
            plt.show()

    def draw(self, scale_x=None, font_size=32, font_color='black', cmap='viridis', filename=None, dpi=960, show=True):
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
        cmap : str, default 'viridis'
            node colormap name, among supported by matplotlib
        filename : str, optional
            file name to save image to
            if None, image is not saved
        dpi : int, default 960
            saved image dpi, higher number mean bigger image
            (matplotlib does not allow to set image size)
        show : bool, default True
            if True, image will be shown

        See Also
        --------
        matplotlib : python data visualisation package
        """
        fig, ax = plt.subplots(1)
        cmap=plt.get_cmap(cmap)
        node_list = sorted(list(self.nodes))
        node_cmap = {node: i/len(node_list) for i, node in enumerate(node_list)}

        subnet_tree = self.subnet_tree()
        subnet_heights = {}

        def calc_height(subnet):
            # also recursevly fills subnet_heights with heights
            inner_keys = set(subnet_tree.nodes[subnet]['inner'])

            if subnet_tree.out_degree(subnet) == 0:
                subnet_heights[subnet] = max(len(inner_keys), subnet_heights.get(subnet,1))
                return subnet_heights[subnet]
            
            key_dict = {child_subnet: data['keys'] for _,child_subnet,data in subnet_tree.out_edges(subnet, data=True)}            
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

            subnet_heights[subnet] = max(height, subnet_heights.get(subnet,1))
            return height

        def node_visual_height(node):
            filtered_subnets = {(s,e): h for (s,e), h in subnet_heights.items() if e==node or s==node}
            return 2*max(filtered_subnets.values(), default=1) - 1

        def edge_visual_length(node, child):
            # get edge length as difference between node and child depth from origin
            child_depth = self.length(end=child)
            node_depth = self.length(end=node)            
            return 2*(child_depth - node_depth) - 1

        def label_center(ox, oy, dx, dy, text):
            centx = (ox + ox + dx) / 2
            centy = (oy + oy + dy) / 2
            plt.text(centx, centy, text, size=font_size/scale_x, va='center', ha='center', color=font_color)

        def depth_to_first_split(start_edge):
            origin, node, _ = start_edge
            s_node = node

            if self.in_degree(node) > 1 or node == self.end:
                return (edge_visual_length(origin, node), node)

            node = next(self.successors(node))
            # while all paths can be traced back to the starting node
            while all(nx.has_path(self,s_node, s) and node!=self.end for s,_ in self.in_edges(node)):
                node = next(self.successors(node))

            return (edge_visual_length(origin, node), node)

        def pop_next_edge(edges_dict, prev_edge):
            # there are 2 key points here - we wanna handle edges in order of their visual length
            # but also we need to ensure that edges that belong to same subnet are drawed in one group
            if prev_edge is None:
                deepest_edge = max(edges_dict, key=lambda elem: edges_dict[elem][0])
                edges_dict.pop(deepest_edge)
                return deepest_edge

            _, prev_split = depth_to_first_split(prev_edge)

            filtered = [(s,e,k) for s,e,k in edges_dict if nx.has_path(self, e, prev_split)]
            if not filtered:
                deepest_edge = max(edges_dict, key=lambda elem: edges_dict[elem][0])
                edges_dict.pop(deepest_edge)
                return deepest_edge

            deepest_edge = max(filtered, key=lambda elem: edges_dict[elem][0])
            edges_dict.pop(deepest_edge)
            return deepest_edge

        calc_height((self.start, self.end))
        origin = (0,0)
        scale_x = scale_x if scale_x is not None else subnet_heights[(self.start, self.end)] / self.length()
        scale_y = 1       
        base_arrow_offset = 0.5 * scale_y
        queue = [(self.start, (0, 0))]
        completed = set()

        while queue:            
            new_nodes = set()
            node, pos = queue.pop(0)

            if node in completed:
                continue

            child_offset = 0
            arrow_offset = base_arrow_offset

            height = scale_y * node_visual_height(node)
            col = cmap(node_cmap[node]); ec = cmap(1-node_cmap[node])
            rect = mpatches.Rectangle(pos, scale_x, height, color=col, ec=ec)
            ax.add_patch(rect)
            label_center(pos[0], pos[1], scale_x, height, str(node))

            new_queue=[]
            edges_dict = {edge: depth_to_first_split(edge) for edge in self.out_edges(node, keys=True)}
            prev_edge = None

            while edges_dict:
                prev_edge = n, child, key = pop_next_edge(edges_dict, prev_edge)
                # child in new nodes means that there are more than 1 edge between node and child
                if child not in new_nodes:
                    arrow_offset = max(child_offset + base_arrow_offset, arrow_offset) 

                alen = scale_x*edge_visual_length(node, child)
                apos = (pos[0]+scale_x, pos[1] + arrow_offset)
                arrow = mpatches.Arrow(apos[0], apos[1], alen, 0, width=0.05/scale_x, color = 'gray')
                ax.add_patch(arrow)
                label_center(apos[0], apos[1], alen, 0, str(key))
                arrow_offset += 2 * scale_y

                if child not in new_nodes and child not in completed:
                    child_height = scale_y * node_visual_height(child)
                    new_nodes.add(child)
                    new_queue.append((child, (pos[0] + alen + scale_x, pos[1] + child_offset)))
                    if edge_visual_length(node, child) == 1:
                        child_offset += child_height + scale_y
                    else:
                        child_offset += 2*scale_y
                elif edge_visual_length(node, child) > 1:
                    child_offset += 2*scale_y

                

            completed.add(node)
            # put newest nodes first - thus makin it kinda like depth-first search
            queue = new_queue + queue

        plt.axis('equal')
        plt.axis('off')

        # removes white padding in the saved image
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

        if show:
            plt.show()
