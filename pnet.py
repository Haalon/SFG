import networkx as nx
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging

from utils import hierarchy_pos

class Pnet(nx.MultiDiGraph):
    """docstring for Pnet"""
    
    def __init__(self, sents=[]):
        super().__init__()
        self.start = 0
        self.end = math.inf
        self.next_node_id = 1

        self.add_sents(sents)

    def add_sents(self, sent_list):
        for sent in sent_list:
            self.add_sent(sent)

    def add_sent(self, sent):
        prev = self.start

        for i, word_or_char in enumerate(sent):
            next_node = self.next_node_by_key(prev, word_or_char)
            if next_node == self.end:
                logging.warning(f"No sentence should be a prefix of another one. Sentence '{sent}' will be ignored")
                return

            if next_node == None:
                self._add_sent(sent[i:], prev)
                return

            prev = next_node

        logging.warning(f"No sentence should be a prefix of another one. Sentence '{sent}' will be ignored")

    def _add_sent(self, sent, start=None):
        prev = start if start is not None else self.start

        for i, word_or_char in enumerate(sent):
            # if we are on the final char - draw edge to the end node of the net
            if i == len(sent) - 1:
                self.add_edge(prev, self.end, label=word_or_char, key=word_or_char)
                break;

            
            self.add_node(self.next_node_id)
            self.add_edge(prev, self.next_node_id, key=word_or_char)
            prev = self.next_node_id

            # just to be sure, may also help later in case of combined p-nets
            while self.next_node_id in self:
                self.next_node_id += 1

    def next_node_by_key(self, node, key):
        if node not in self:
            return None

        # edge[2] contains key
        labeled_edges = [edge for edge in self.out_edges(node, keys=True) if edge[2] == key]

        if len(labeled_edges) == 0:
            return None

        if len(labeled_edges) > 1:
            raise ValueError("Net should not have outgoing edges starting with same symbol")

        return labeled_edges[0][1]

    def is_transit_node(self, node):
        return self.out_degree(node) == self.in_degree(node) == 1

    def terminals(self):
        return set(nx.get_edge_attributes(self,'label').values())

    def height(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        return len(list(nx.all_simple_edge_paths(self, start, end)))

    def length(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end
        return len(max(list(nx.all_simple_edge_paths(self, start, end)), key=len, default=[]))

    def get_sents(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        paths = nx.all_simple_edge_paths(self, start, end)
        res = []
        for path in paths:
            sent = ''
            for edge in path:
                sent += self.edges[edge][2]
            res.append(sent)

        return res


    def subnet(self, start=None, end=None):
        start = start if start is not None else self.start
        end = end if end is not None else self.end

        desc = nx.descendants(self, start)
        desc.add(start)

        if end not in desc:
            return None

        ancs = nx.ancestors(self, end)
        ancs.add(end)
        new_net = self.subgraph(desc.intersection(ancs))

        nx.relabel_nodes(new_net, {end: math.inf}, copy=False)
        new_net.start = start
        new_net.end = math.inf
        new_net.next_node_id = end
        return new_net

    def in_subnet(self, subnet, node):
        s_start, s_end = subnet

        if s_start == node or s_end == node:
            return True

        desc = nx.descendants(self, s_start)
        ancs = nx.ancestors(self, s_end)

        return node in desc and node in ancs

    def subnet_list(self):
        start_nodes = {node: self.out_degree(node) for node in self.nodes() if self.out_degree(node) > 1 or node == self.start}

        subnets = []

        for node, out_degree in start_nodes.items():
            for i, path_node in enumerate(nx.shortest_path(self, node, self.end)):
                if path_node != node and self.in_degree(path_node) >= out_degree:
                    subnets.append((node, path_node))
                    break

        return subnets

    def subnet_tree(self, subnet_list=None):
        subnet_list = subnet_list if subnet_list is not None else self.subnet_list()
        subnet_lens = {(s,e): self.length(s,e) for s,e in subnet_list}

        tree = nx.DiGraph()

        for s,e in subnet_list:
            enveloping_subnets = [subnet for subnet in subnet_list if self.in_subnet(subnet, s) and self.in_subnet(subnet, e)]
            enveloping_subnets.remove((s, e))

            min_enveloping_subnet = min(enveloping_subnets, default=None, key = lambda subnet: subnet_lens[subnet])
            
            if min_enveloping_subnet is None:
                tree.add_node((s,e))
            else:
                tree.add_edge(min_enveloping_subnet, (s,e))

        return tree

    def draw_subnet_tree(self, subnet_tree=None, cmap='viridis', filename=None, dpi=960, show=True, **kwargs):
        subnet_tree = subnet_tree if subnet_tree is not None else self.subnet_tree()
        pos = hierarchy_pos(subnet_tree,(self.start, self.end))

        node_list = sorted(list(self.nodes))
        node_cmap = [i/len(node_list) for i, _ in enumerate(node_list)]
        tree_cmap = [col for i, col in enumerate(node_cmap) if self.out_degree(node_list[i]) > 1 or node_list[i] == self.start]

        fig = plt.figure()
        nx.draw(subnet_tree, ax=fig.add_subplot(111), pos=pos, with_labels=True, node_color=tree_cmap, cmap=cmap, **kwargs)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi, pad_inches=0)

        if show:
            plt.show()

    def draw(self, scale_x=None, font_size=32, font_color='black', cmap='viridis', filename=None, dpi=960, show=True):
        fig, ax = plt.subplots(1)

        origin = (0,0)
        scale_x = scale_x if scale_x is not None else self.height() / self.length()
        scale_y = 1       
        base_arrow_offset = 0.5 * scale_y

        cmap=plt.get_cmap(cmap)
        node_list = sorted(list(self.nodes))
        node_cmap = {node: i/len(node_list) for i, node in enumerate(node_list)}

        def label_center(ox, oy, dx, dy, text):
            centx = (ox + ox + dx) / 2
            centy = (oy + oy + dy) / 2
            plt.text(centx, centy, text, size=font_size/scale_x, va='center', ha='center', color=font_color)

        def node_visual_height(node):
            in_count = self.in_degree(node)
            height = self.height(node)
            return 2*max(height, in_count) - 1

        # get edge length as difference between node and child depth from origin
        def edge_visual_length(node, child):
            child_depth = self.length(end=child)
            node_depth = self.length(end=node)            
            return 2*(child_depth - node_depth) - 1

        total_length = 2*self.length()*scale_x
        queue = {self.end: (total_length, 0), self.start: (0, 0)}
        completed = set()

        while queue:
            new_queue = {}
            for node, pos in queue.items():
                if node in completed:
                    continue

                child_offset = 0
                arrow_offset = base_arrow_offset

                height = scale_y * node_visual_height(node)
                col = cmap(node_cmap[node]); ec = cmap(1-node_cmap[node])
                rect = mpatches.Rectangle(pos, scale_x, height, color=col, ec=ec)
                ax.add_patch(rect)
                label_center(pos[0], pos[1], scale_x, height, str(node))

                for _, child, key, data in self.out_edges(node, keys=True, data=True):
                    alen = scale_x*edge_visual_length(node, child)
                    apos = (pos[0]+scale_x, pos[1] + arrow_offset)
                    arrow = mpatches.Arrow(apos[0], apos[1], alen, 0, width=0.05/scale_x, color = 'gray')
                    ax.add_patch(arrow)
                    label_center(apos[0], apos[1], alen, 0, str(key))
                    arrow_offset += 2 * scale_y

                    if child not in new_queue and child not in completed:
                        child_height = scale_y * node_visual_height(child)
                        # 2 should be replaced with arrow length + scale_x
                        new_queue[child] = (pos[0] + alen + scale_x, pos[1] + child_offset)
                        child_offset += child_height + scale_y
                        arrow_offset = child_offset + base_arrow_offset
                    else:
                        child_offset += 2*scale_y

                completed.add(node)

            queue = new_queue.copy()

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
