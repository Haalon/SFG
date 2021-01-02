import networkx as nx
import math 
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout


class PNet(nx.DiGraph):
	"""docstring for PNet"""

	# should not be changed
	_start = 0
	_end = math.inf
	
	def __init__(self, sent_list):
		super().__init__()
		
		self.add_node(PNet._end, label=PNet._end)
		self.add_node(PNet._start, label=PNet._start)
		curr = PNet._start

		terminals = set()

		for sent in sent_list:
			prev = PNet._start
			for i, char in enumerate(sent):
				terminals.add(char)
				if i == len(sent) - 1:
					self.add_edge(curr, PNet._end, label=char)
					break;

				curr += 1
				self.add_node(curr, label=char)
				self.add_edge(prev, curr, label=char)
				prev = curr

		self.count = curr
		self.terminals = terminals

	def draw(self):
		pos = nx.spring_layout(self)
		print(pos)
		# pos =graphviz_layout(self, prog='dot')

		labels = nx.get_edge_attributes(self,'label')
		node_labels = nx.get_node_attributes(self, 'label')

		terminals = list(self.terminals)

		val_map = {term: terminals.index(term) / len(terminals) for term in terminals}
		values = [val_map.get(labl, 0.25) for labl in node_labels.values()]


		nx.draw_networkx_nodes(self, pos, cmap=plt.get_cmap('viridis'), node_color=values)
		nx.draw_networkx_edges(self, pos, edge_color='gray')

		# nx.draw_networkx_labels(self, pos, font_size=8, font_family='sans-serif')
		nx.draw_networkx_edge_labels(self,pos, edge_labels=labels)

		plt.axis('off')
		plt.show()


				


