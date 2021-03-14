from nltk import CFG, Nonterminal, Production


def terminals(g):
	res = set()
	for prod in g.productions():
		for item in prod.rhs():
			if not isinstance(item, Nonterminal):
				res.add(item)

	return res

def nonterminals(g):
	res = set()
	for prod in g.productions():
		nont = prod.lhs()
		res.add(nont)

	return res

def children(g, parent):
	res = set()

	if isinstance(parent, Production):
		prods = [parent]
	else:
		prods = g.productions(parent)

	for prod in prods:
		for item in prod.rhs():
				if isinstance(item, Nonterminal):
					res.add(item)

	return res

def endings(g, n):
	res = set()

	for prod in g.productions(n):
		if all(not isinstance(item, Nonterminal) for item in prod.rhs()):
			res.add(prod.rhs())

def is_separated(g):
	nonts = nonterminals(g)

	for nont in nonts:
		starts = set()
		for prod in g.productions(nont):
			start = prod.rhs()[0]

			if isinstance(start, Nonterminal):
				return False

			if start in starts:
				return False

			starts.add(start)
 
	return True

def unreachable(g):
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


# def is_canonical(g):
# 	if not is_separated(g):
# 		raise ValueError("Non-separated grammar was given")

# 	nonts = nonterminals(g)

# 	broken_rules = set()

# 	ends = {nont: set() for nont in nonts}
# 	counts = {nont: 0 for nont in nonts}

# 	for prod in g.productions():
# 		ends[prod.lhs()].add(prod.rhs()[-1])
# 		counts[prod.lhs()] += 1

# 		for item in prod:
# 			if item == grammar.start():
# 				broken_rules.add(1)
 
# 	return True
