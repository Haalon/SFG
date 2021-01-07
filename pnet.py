import networkx as nx
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


    def get_next(self, node, word_or_char):
        labeled_edges = [edge for edge in self.out_edges(node, keys=True) if edge[2] == word_or_char]

        if len(labeled_edges) == 0:
            return None

        if len(labeled_edges) > 1:
            raise ValueError("Net should not have outgoing edges starting with same symbol")

        return labeled_edges[0][1]


    def merge_sent(self, sent):
        prev = Pnet._start

        for i, word_or_char in enumerate(sent):
            next_node = self.get_next(prev, word_or_char)
            if next_node == None:
                self.add_sent(sent[i:], prev)
                return

            prev = next_node

        raise ValueError("No sent should be a prefix of another sentence")

    def add_sent(self, sent, start=None):
        prev = start if start is not None else Pnet._start

        for i, word_or_char in enumerate(sent):
            # if we are on the final char - draw edge to the end polus of the net
            if i == len(sent) - 1:
                self.add_edge(prev, Pnet._end, label=word_or_char, key=word_or_char)
                break;

            self.node_id += 1
            self.add_node(self.node_id, label=word_or_char)
            self.add_edge(prev, self.node_id, label=word_or_char, key=word_or_char)
            prev = self.node_id

    def is_transit_node(self, node):
        return self.out_degree(node) == self.in_degree(node) == 1

    def subnet(self, start, end):
        desc = nx.descendants(self, start)
        if end not in desc:
            return None

        ancs = nx.ancestors(self, end)

        return self.subgraph(desc).subgraph(ancs)


    def terminals(self):
        return set(nx.get_edge_attributes(self,'label').values())

    def height(self, start=None, end=None):
        start = start if start is not None else Pnet._start
        end = end if end is not None else Pnet._end
        return len(list(nx.all_simple_edge_paths(self, start, end)))

    def length(self, start=None, end=None):
        start = start if start is not None else Pnet._start
        end = end if end is not None else Pnet._end
        return len(max(list(nx.all_simple_edge_paths(self, start, end)), key=len, default=[]))

    def get_sents(self):
        paths = nx.all_simple_edge_paths(self, Pnet._start, Pnet._end)
        res = []
        for path in paths:
            sent = ''
            for edge in path:
                sent += self.edges[edge]['label']
            res.append(sent)

        return res

    def draw(self, scale_x=None, font_size=32, filename=None, dpi=960, show=True):
        fig, ax = plt.subplots(1)

        origin = (0,0)
        scale_x = scale_x if scale_x is not None else self.height() / self.length()
        scale_y = 1       
        base_arrow_offset = 0.5 * scale_y

        cmap=plt.get_cmap('viridis')
        node_list = sorted(list(self.nodes))
        node_cmap = {node: i/len(node_list) for i, node in enumerate(node_list)}

        def label_center(ox, oy, dx, dy, text):
            centx = (ox + ox + dx) / 2
            centy = (oy + oy + dy) / 2
            plt.text(centx, centy, text, size=font_size/scale_x, va='center', ha='center')

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
        queue = {Pnet._end: (total_length, 0), Pnet._start: (0, 0)}
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
