import networkx as nx
import math 
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout


class Pnet(nx.MultiDiGraph):
    """docstring for Pnet"""

    # should not be changed
    _start = 0
    _end = math.inf
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_id = Pnet._start

        # added manually just for the labels
        self.add_node(Pnet._start, label=Pnet._start)
        self.add_node(Pnet._end, label=Pnet._end)
        

    def add_sents(self, sent_list):
        for sent in sent_list:
            self.merge_sent(sent)


    def get_next(self, node, char):
        labeled_edges = [edge for edge in self.out_edges(node, keys=True) if edge[2].startswith(char)]

        if len(labeled_edges) == 0:
            return None

        if len(labeled_edges) > 1:
            raise ValueError("Net should not have outgoing edges starting with same symbol")

        return labeled_edges[0][1]


    def merge_sent(self, sent):
        prev = Pnet._start

        for i, char in enumerate(sent):
            next_node = self.get_next(prev, char)
            if next_node == None:
                self.add_sent(sent[i:], prev)
                return

            prev = next_node

        raise ValueError("No sent should be a prefix of another sentence")

    def add_sent(self, sent, start=None):
        prev = start if start is not None else Pnet._start

        for i, char in enumerate(sent):
            # if we are on the final char - draw edge to the end polus of the net
            if i == len(sent) - 1:
                self.add_edge(prev, Pnet._end, label=char, key=char)
                break;

            self.node_id += 1
            self.add_node(self.node_id, label=char)
            self.add_edge(prev, self.node_id, label=char, key=char)
            prev = self.node_id

    def terminals(self):
        return set(nx.get_edge_attributes(self,'label').values())

    def width(self, start=None, end=None):
        start = start if start is not None else Pnet._start
        start = end if end is not None else Pnet._end
        return len(list(nx.all_simple_edge_paths(self, start, end)))

    def length(self, start=None, end=None):
        start = start if start is not None else Pnet._start
        start = end if end is not None else Pnet._end
        return len(max(list(nx.all_simple_edge_paths(self, start, end)), key = len))

    def get_sents(self):
        paths = nx.all_simple_edge_paths(self, Pnet._start, Pnet._end)
        res = []
        for path in paths:
            sent = ''
            for edge in path:
                sent += self.edges[edge]['label']
            res.append(sent)

        return res

    def draw(self):
        pos = nx.spring_layout(self)
        print(pos)
        # pos =graphviz_layout(self, prog='dot')

        labels = nx.get_edge_attributes(self,'label')
        node_labels = nx.get_node_attributes(self, 'label')

        terminals = list(self.terminals())

        val_map = {term: terminals.index(term) / len(terminals) for term in terminals}
        values = [val_map.get(labl, 0.25) for labl in node_labels.values()]


        nx.draw_networkx_nodes(self, pos, cmap=plt.get_cmap('viridis'), node_color=values)
        nx.draw_networkx_edges(self, pos, edge_color='gray')

        # nx.draw_networkx_labels(self, pos, font_size=8, font_family='sans-serif')
        nx.draw_networkx_edge_labels(self,pos, edge_labels=labels)

        plt.axis('off')
        plt.show()
