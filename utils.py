import networkx as nx
import random


def merge_nodes_and_keys(G, keep_node, merge_nodes):
    """Merges nodes into one, while also merging edges with same key

    Parameters
    ----------    
    G : networkx.MultiDigraph
        the graph
    keep_node : graph node
        node that all others will be merged into
    merge_nodes : list of graph nodes
        nodes to merge
    """
    out_keys = {key for node in merge_nodes for _,_,key in G.out_edges(node,keys=True)}
    in_keys = {key for node in merge_nodes for _,_,key in G.in_edges(node,keys=True)}

    out_keys.update(k for _,_,k in G.out_edges(keep_node,keys=True))
    in_keys.update(k for _,_,k in G.in_edges(keep_node,keys=True))

    nx.relabel_nodes(G, {m_node: keep_node for m_node in merge_nodes}, copy=False)

    # relabeling in netwrokx may create new unwanted edges automatically
    for (s,e,k) in list(G.out_edges(keep_node, keys=True)):
        if k not in out_keys:
            G.remove_edge(s,e,k)

    for (s,e,k) in list(G.in_edges(keep_node, keys=True)):
        if k not in in_keys:
            G.remove_edge(s,e,k)


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """If the graph is a tree this will return the positions to plot this in a hierarchical layout

    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike     
     
    Parameters
    ----------    
    G : networkx.Digraph
        the graph (must be a tree)
    
    root
        the root node of current branch

        * if the tree is directed and this is not given, 
          the root will be found and used
        * if the tree is directed and this is given, then 
          the positions will be just for the descendants of this node.
        * if the tree is undirected and not given, 
          then a random choice will be used.
    
    width : float, default 1
        horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap : float, default 0.2
        gap between levels of hierarchy
    
    vert_loc: float, default 0
        vertical location of root
    
    xcenter: float, default 0.5
        horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)