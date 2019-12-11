"""
Script for MCS+
Reliable Query Response
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os

from my_community import mycommunity
from multi_arm_bandit import bandit

import networkx as nx
import community
import csv
import numpy as np
import random
import pickle
import operator
import operator
import traceback
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import LinearRegression
import time
import multiprocessing
from multiprocessing import Pool

ctheta = 2


def getApproxPartition(graph, nodes=None, single=True):
	if graph.number_of_edges() == 0:
		return {u:u for u in graph.nodes()}, 0

	# Replace with other community detection algorithm if desired
	part = community.best_partition(graph)
	mod = community.modularity(part, graph)

	return part, mod


class LayerImportance(object):
	"""Class for handling layer importance realated methods"""
	def __init__(self, layer_count):
		super(LayerImportance, self).__init__()
		self._overlap = [[0.0] for _ in range(0,layer_count)]					# importances of layers
		self._freshness = [[1.0] for _ in range(0, layer_count)]					# Amount of new edges found in previous round

	def _updateBaseGraph(self, graph, nodes=None):
		"""
		The graph against which importance will be calculated
		"""
		self._base_graph = graph 													# graph on which importances calulations well be based on
		
		if nodes is not None:
			self._base_graph = self._base_graph.subgraph(nodes)

		self._base_nodes = set(list(self._base_graph.nodes()))
		self._base_edges = set([frozenset(e) for e in self._base_graph.edges()])


	def _edgeOverlap(self, graph):
		"""
		Fraction of edges in graph that are also in base graph
		If nodes is None, all the nodes in graph are considered.
		Otherwise only subgraph containing nodes is considered.
		"""
		sg = graph.subgraph(self._base_nodes)

		if sg.number_of_edges() == 0:
			# If there are no edges in subgraph, return False
			return 0.0

		edges = set([frozenset(e) for e in sg.edges()])
		
		return len(self._base_edges.intersection(edges))/len(edges)


	def _randomEdgeOverLap(self, graph):
		"""
		Expected fraction of overlap if graph were random
		"""
		sg = graph.subgraph(self._base_nodes)

		if sg.number_of_edges() == 0:
			# If there are no edges in subgraph, return False
			return 0.0, 1.0

		# Edge probality for random graph based on graph
		ep = 2 * sg.number_of_edges() / (sg.number_of_nodes())**2
		
		# Number of edges between nodes in base graph
		base_edge_count = self._base_graph.subgraph(sg.nodes()).number_of_edges()

		# Expected number of overlap edge
		mu = base_edge_count * ep
		var = np.sqrt(base_edge_count * ep * (1.0 - mu)**2)

		# Overlap edges as fraction of all edges in sg
		#print(mu, var)
		return mu, var


	def _computeOverlap(self, graph):
		"""
		Compute the relative layer importance
		"""
		val = self._edgeOverlap(graph)

		mu, var = self._randomEdgeOverLap(graph)

		if var == 0:
			i = 0.0
		else:
			i = np.abs((val - mu)/var)

		return max(i, 0.0)


	def updateLayerOverlap(self, graphs, nodes=None):
		"""
		Update the importance of all layers in graphs, and nodes

		"""
		self._updateBaseGraph(graphs[0], nodes)

		for i in range(0, len(graphs)):
			overlap = self._computeOverlap(graphs[i])
			if overlap is not False:
				self._overlap[i].append(overlap)


	def updateLayerFreshness(self, i, val):
		self._freshness[i].append(val)


	def getLayerFreshness(self, layer=None):
		# Freshness of the last 5 rounds
		if layer is not None:
			return np.mean(self._freshness[layer][-3:])
		else:
			return[np.mean(f[-3:]) for f in self._freshness]


	def getLayerOverlap(self, layer=None):
		if layer is not None:
			return self._overlap[layer][-1]
		else:
			return [i[-1] for i in self._overlap]
					

class Budget(object):
	"""docstring for Budget"""
	def __init__(self, max_budget, layer_costs, layer_importance):
		super(Budget, self).__init__()
		self._budget_max = max_budget
		self._budget_left = max_budget
		self._budget_consumed = 0
		self._layer_costs = layer_costs
		self._slices = 10											# Initial number of slices
		self._slices_last_update = 0 								# The budget consumed when slice was last updated
		self._layer_importance = layer_importance


	def initializeBudget(self):
		"""
		Allocate 10% of max budget to first slice
		Allocate enough budget such that same number of queries can 
		be made in each layer
		"""
		budget = self._budget_left/self._slices
		total_cost = sum(self._layer_costs)
		allocated = []

		for c in self._layer_costs:
			allocated.append(budget * c / total_cost)

		return allocated


	def consumeBudget(self, cost):
		self._budget_consumed += cost
		self._budget_left -= cost


	def updateSlices(self):
		"""
		Update number of slices based on cost consumed since last update
		"""
		if self._budget_consumed == self._slices_last_update:
			return True
		cost = self._budget_consumed - self._slices_last_update
		self._slices = min(self._slices, np.ceil(self._budget_left / cost))
		self._slices = max(self._slices, 1)
		self._slices_last_update = self._budget_consumed


	def allocateBudget(self):
		"""
		Allocate the budget based on weights
		Layers with high weight gets more budget
		Budget for layer 0 depends only on layer cost
		"""
		budget = self._budget_left/self._slices
		allocation = []

		# Budget for layer 0 
		b0 = budget * self._layer_costs[0] / np.sum(self._layer_costs)
		allocation.append(b0)
		n0 = b0 / self._layer_costs[0]

		# Remainig budget
		budget -= b0

		# Total weights excluding layer 0
		eta = 0.000000001
		weights = [self._layer_importance.getLayerOverlap(l) * self._layer_importance.getLayerFreshness(l) for l in range(1, len(self._layer_costs))]
		total_weight = np.sum(weights) + eta * len(weights) 

		for i in range(0, len(weights)):
			b = budget * (weights[i] + eta) / total_weight
			b = min(b, n0 * self._layer_costs[i+1])
			allocation.append(b)

		# Make sure each layer get enough for at least one query
		allocation = [max(allocation[i], self._layer_costs[i]) for i in range(0, len(allocation))]
		
		return allocation

	def getBudgetLeft(self):
		return self._budget_left

	def getBudgetConsumed(self):
		return self._budget_consumed
		

class Evaluation(object):
	"""docstring for Evaluation"""
	def __init__(self, graphs, partition=None):
		super(Evaluation, self).__init__()
		self._graphs = graphs

		# Partitions and communties of full layer 0
		if partition is None:
			self._partition = self._getPartition(self._graphs[0])
		else:
			self._partition = partition

		self._community = self._getCommunity(self._partition)
		self._partition = self._communityToPartition(self._community)
		
		self._cweights = {i:len(self._community[i])/len(self._partition)\
		 for i in self._community} 					# Relative size of the communtities


	def _getPartition(self, graph):
		return community.best_partition(graph, randomize=False)


	def _communityToPartition(self, com):
		part = {}
		for c in com:
			for u in com[c]:
				part[u] = c
		return part

	def _getCommunity(self, partition):
		com = {}

		for n in partition:
			p = partition[n]
			if p not in com:
				com[p] = set()
			com[p].update(set([n]))

		#com = {c:com[c] for c in com if len(com[c]) > 1}

		# Make sure we do not consider the singleton nodes
		return com


	def _communityQuality(self, x0, x1):
		return normalized_mutual_info_score(x0, x1)


	def _communityRepresentation(self, com):
		m0, m1, s0, s1, eta = 0, 0, 0, 0, 0

		com0 = list(self._community.values())
		com0 = sorted(com0, key=len, reverse=True)

		com1 = list(com.values())
		com1 = sorted(com1, key=len, reverse=False)

		for i in range(0, len(com0)):
			max_sim = 0
			max_com = None
			
			for j in range(0, len(com1)):
				sim = len(com1[j].intersection(com0[i]))
				if sim > max_sim:
					max_sim = sim
					max_com = j
			if max_com is not None:
				#com1.pop(max_com)
				m0 += np.log10(len(com0[i]) + eta)
				#m0 += 1
				#break
		"""
		for i in range(0, len(com1)):
			#max_sim = 0
			#max_com = None
			
			for j in range(0, len(com0)):
				sim = len(com0[j].intersection(com1[i]))
				#if sim > max_sim:
				#	max_sim = sim
				#	max_com = j
				if sim > 0:
					m1 += np.log10(len(com1[i]) + eta)
					break
		"""
		#c0 = len(com0)
		#print([np.log10(len(c) + eta) for c in com1])
		c0 = np.sum([np.log10(len(c) + eta) for c in com0])
		#c1 = np.sum([np.log10(len(c) + eta) for c in com1])

		if c0 == 0:
			return 0.0

		return m0 / c0

		s0 = m0 / c0
		s1 = m1 / c1

		#print(s0, s1)

		cr = 2 * s0 * s1 / (s0 + s1)

		return s0


	def communitySimilarity(self, graph, nodes=None):
		if graph.number_of_edges() == 0:
			return [0,0,0]

		part, _ = getApproxPartition(graph, nodes)
		#nodes = graph.nodes()
		"""
		if nodes is None:
		#	#part = self._getPartition(graph)
			part, _ = getApproxPartition(graph)
		else:
			sg = graph.subgraph(nodes)
			if sg.number_of_edges() == 0:
				return [0,0,0]
			#part = self._getPartition(sg)
			part, _ = getApproxPartition(sg)
		"""
		# Common nodes to perform comparison
		part = {u:part[u] for u in nodes}
		nodes = set(part.keys()).intersection(self._partition.keys())

		#nodes = nodes.intersection(nodes0)
		#if nodes is not None and len(nodes) > 0:
		#part = {u:part[u] for u in part if u in nodes}
		#el
		#	return 0.0

		com = self._getCommunity(part)

		x0 = [self._partition[u] for u in nodes]
		x1 = [part[u] for u in nodes]

		#print(x0, x1)

		q = self._communityQuality(x0, x1)
		r = self._communityRepresentation(com)

		#print(q,r)

		if r + q == 0:
			return [0,0,0]

		return [2 * q * r / (q + r), q, r]


	def partitionDistance(self, part1, part2, nodes=None):
		"""
		Compute the partiton distance between communities c1 and c2
		"""

		c1 = self._getCommunity(part1)
		c2 = self._getCommunity(part2)

		if nodes is None:
			n1 = set([])
			n2 = set([])
			for c in c1:
				n1.update(c1[c])
			for c in c2:
				n2.update(c2[c])
			nodes = n1.intersection(n2)

		c1 = {c:c1[c].intersection(nodes) for c in c1}
		c2 = {c:c2[c].intersection(nodes) for c in c2}

		m = max(len(c1), len(c2))
		m = range(0,m)

		mat = {i: {j: 0 for j in c2} for i in c1}

		total = 0
		for i in c1:
			for j in c2:
				if i in c1 and j in c2:
					mat[i][j] = len(c1[i].intersection(c2[j]))
					total += mat[i][j]

		if total <= 1:
			return 1.0

		assignment = []
		rows = c1.keys()
		cols = c2.keys()

		
		while len(rows) > 0 and len(cols) > 0:
			mval = 0
			r = -1
			c = -1
			for i in rows:
				for j in cols:
					if mat[i][j] >= mval:
						mval = mat[i][j]
						r = i 
						c = j
			rows.remove(r)
			cols.remove(c)
			assignment.append(mval)
		
		dist = total - np.sum(assignment)

		if np.isnan(dist/total):
			return 0
			
		return dist/total


class NodeSelection(object):
	"""docstring for NodeSelection"""
	def __init__(self, sample):
		super(NodeSelection, self).__init__()
		self._sample = sample
		self._model = None
		self._tdata = {'X':[], 'Y':[]}
		self._alpha = 0.1					# probability of selecting random node
		self._lfeatures = None


	def _getFeatures(self, candidates):
		degree = nx.degree_centrality(self._sample)
		betweeness = nx.betweenness_centrality(self._sample, k=min(10, self._sample.number_of_nodes()))
		core = nx.core_number(self._sample)

		# Normalize all features between 0 and 1
		min_degree, max_degree = min(degree.values()), max(degree.values())
		min_betweeness, max_betweeness = min(betweeness.values()), max(betweeness.values())
		min_core, max_core = min(core.values()), max(core.values())

		vdegree = {u:0 for u in candidates}
		vbetweeness = {u:0 for u in candidates}
		vcore = {u:0 for u in candidates}

		if min_degree < max_degree:
			vdegree.update({u: (degree[u] - min_degree)/(max_degree - min_degree) for u in degree})
		
		if min_betweeness < max_betweeness:
			vbetweeness.update({u: (betweeness[u] - min_betweeness)/(max_betweeness - min_betweeness) for u in betweeness})
		
		if min_core < max_core:
			vcore.update({u: (core[u] - min_core)/(max_core - min_core) for u in core})

		features = [[vdegree[u], vbetweeness[u], vcore[u]] for u in candidates]

		return features


	def nextNode(self, candidates):
		if len(candidates) == 0:
			self._lfeatures = None
			return False

		candidates = list(candidates)
		features = self._getFeatures(candidates)

		if np.random.random() < self._alpha or self._model is None or len(self._tdata['X']) < 5:
			m_index = np.random.choice(len(candidates))
		else:
			Y = self._model.predict(features)
			m_index, m_val = -1, -10000
			for i in range(0, len(Y)):
				if Y[i] > m_val:
					m_val = Y[i]
					m_index = i
		self._lfeatures = features[m_index]
		return [candidates[m_index]]


	def update(self, y, sample):
		self._sample = sample
		if self._lfeatures is not None:
			self._tdata['X'].append(self._lfeatures)
			self._tdata['Y'].append(y)
			self._model = LinearRegression().fit(self._tdata['X'], self._tdata['Y'])


class RNDSample(object):
	"""docstring for RNDSample"""
	def __init__(self, graph, sample, layer_costs, queried, budget, layer_importance):
		super(RNDSample, self).__init__()
		self._sample = sample
		self._graph = graph
		self._layer_costs = layer_costs
		self._queried = queried
		self._unqueried = [set([]) for _ in self._sample]
		self._alpha = 0.1 									# reset prob for random walk
		self._budget = budget
		self._layer_importance = layer_importance

		self._initializeSample()


	def _initializeSample(self):
		"""
		Initialize sample by adding some random nodes to samples
		"""
		
		nodes = sorted(list(self._graph[0].nodes()))[:10]
		for i in range(0, len(self._sample)):
			self._sample[i].add_nodes_from(nodes)
			self._unqueried[i].update(nodes)


	def sample(self, budget):
		"""
		Sample graph with random walk
		"""
		for i in range(0, len(self._sample)):
			self._unqueried[i].difference_update(self._queried[i])
		
			if len(self._unqueried[i]) > 0:
				u = np.random.choice(list(self._unqueried[i]))
			else:
				l = np.random.choice(range(0, len(self._unqueried)))
				if len(self._unqueried[l]) > 0:
					u = np.random.choice(list(self._unqueried[l]))
				else:
					u = None
			c = 0
			edges0 = set([frozenset(e) for e in self._sample[i].edges()])

			while c <= budget[i] and u is not None and self._budget.getBudgetLeft() > 0:
				c += self._layer_costs[i]
				self._budget.consumeBudget(self._layer_costs[i])

				try:
					neighbors = set(list(self._graph[i].neighbors(u)))
					edges = [(u,v) for v in neighbors]
					self._sample[i].add_edges_from(edges)
				except:
					neighbors = []

				self._queried[i].update([u])
				self._unqueried[i].update(neighbors)
				self._unqueried[i].difference_update(self._queried[i])
				
				# If no unqueried node, stop
				if len(self._unqueried[i]) == 0:
					break

				candidates = set(neighbors).difference(self._queried[i])
				
				if np.random.random_sample() > self._alpha and len(candidates) > 0:
					u = np.random.choice(list(candidates))
				elif len(self._unqueried[i]) > 0:
					u = np.random.choice(list(self._unqueried[i]))
				else:
					break

			# Update layer importance
			freshness = 0
			if self._sample[i].number_of_edges() > 0:
				edges1 = set([frozenset(e) for e in self._sample[i].edges()])
				freshness = len(edges1.difference(edges0)) / len(edges1)

			self._layer_importance.updateLayerFreshness(i, freshness)


class CommunityManager(object):
	"""docstring for CBanditManager"""
	def __init__(self, hcommunity):
		super(CommunityManager, self).__init__()
		self._hcommunity = hcommunity
		self._initalCommunities()
		self._generateMapping()

	def _generateMapping(self):
		"""
		Map int to com ids
		"""
		for l in  range(0,self._hcommunity.getLayerCount()):
			c = self._hcommunity.getCommunityIds(l)
			m = {i:c[i] for i in range(0, len(c))}
			r = {c[i]:i for i in range(0, len(c))}


	def _getComName(self, layer, i):
		"""
		Return com name given layer and id
		"""
		return self._map[layer][i]


	def _initalCommunities(self):
		"""
		The two initial communities for all layers
		"""
		roots = self._hcommunity.getRootCommunity()
		self._active_communities = []
		self._rewards = []
		self._crewards = []
		
		for l in range(0, self._hcommunity.getLayerCount()):
			coms = self._hcommunity.getChildren(l, roots[l])
			self._active_communities.append(coms)
			self._crewards.append({c:[] for c in coms})


	def getActiveCommunities(self, layer):
		return self._active_communities[layer]


	def updateCReward(self, layer, cid, value):
		#cid = self._map[layer][cid]
		self._rewards.append(value)
		self._crewards[layer][cid].append(value)


	def switchArm(self, layer):
		"""
		Check rewards to check if active community need to be changed
		"""
		if np.any([len(self._crewards[layer][l]) for l in self._crewards[layer]] < 5) :
			return False

		rewards = self._crewards[layer]
		cid = self._active_communities[layer]

		aval = np.mean(self._rewards)
		astd = np.std(self._rewards)

		mval = {c:np.mean(rewards[c]) for c in cid}
		sval = {c:np.std(rewards[c]) for c in cid}

		changed = False

		# If both arms have very low rewards, swith up
		if np.all([mval[c] + sval[c] for c in cid] < aval):
			self.switchArmUp(layer, cid[0])
		elif mval[cid[0]] < mval[cid[1]] - sval[cid[1]]:
			# If arm 0 is very much lower than 1, swith down to 1
			self.switchArmDown(layer, cid[1])
		elif mval[cid[1]] < mval[cid[0]] - sval[cid[0]]:
			self.switchArmDown(layer, cdi[0])

		if changed:
			cid = self._active_communities[layer]
			self._crewards[layer] = {c:[] for c in cid}


	def switchArmDown(self, layer, cid):
		"""
		Switch to a lower level of community from comid
		"""
		active = self.getActiveCommunities(layer)

		if comid not in active:
			return None

		if self._hcommunity.checkLeaf(layer, cid):
			# If leaf, return cid and sibling
			return (cid, self._hcommunity.getSibling(layer, cid)[0])

		return self._hcommunity.getChildren(layer, cid)


	def switchArmUp(self, layer, cid):
		active = self.getActiveCommunities(layer)

		if comid not in active:
			return None

		parent = self._hcommunity.getParent(layer, cid)
		
		if self._hcommunity.checkLeaf(layer, parent):
			# if parent is root, return self and sibling
			return (cid, self._hcommunity.getSibling(layer, cid)[0])

		return (parent, self._hcommunity.getSibling(layer, parent)[0])


class BanditManager(object):
	"""Manages the multiple bandits"""
	def __init__(self, graph, sample, queried, commode=0, bmode=1):
		super(BanditManager, self).__init__()
		self._bmode = bmode
		self._epsilon = 0.2
		
		self._graph = graph
		self._sample = sample
		self._queried = queried
		self._layers = range(0, len(graph))

		self._lbandit = bandit.MultiArmBandit(self._layers,\
		 mode=self._bmode, epsilon=self._epsilon)
		self._cbandit = [None for _ in self._layers]
		self._rbandit = [NodeSelection(self._sample[i]) for i in range(0, len(self._sample))]

		self._commode = commode


	def getLayerEstimates(self):
		# Returns the estimated rewards for all the layers
		return self._lbandit.getEst()

	def initializeCBandits(self):
		self._hcommunity = CommunitHeirarchy(self._sample, mode=self._commode)
	
		if self._commode == 0:
			self._commanager = CommunityManager(self._hcommunity)
			self._cbandit = [bandit.MultiArmBandit(self._hcommunity.getCommunityIds(l).values(),\
			 mode=self._bmode, epsilon=self._epsilon) for l in self._layers]
		else:
			self._commanager = None
			self._cbandit = [bandit.MultiArmBandit(self._hcommunity.getCommunityIds(l), mode=self._bmode, epsilon=self._epsilon) for l in self._layers]
		


	def _nextArms(self):
		"""
		Get the next layer and role
		"""
		#try:
		larm, _ = self._lbandit.nextArm()

		if self._commode == 0:
			carm, _ = self._cbandit[larm].nextArm(arms=self._commanager.getActiveCommunities(larm))
		else:
			carm, _ = self._cbandit[larm].nextArm()

		self._arm = [larm, carm, -1]


	def getArms(self):
		return self._arm


	def updateReward(self, reward, sample):
		self._lbandit.updateEst(reward[0], self._arm[0])
		self._rbandit[self._arm[0]].update(reward[2], sample[self._arm[0]])

		if self._commode == 0:
			self._updateRewardCBandit(reward[1], True)
			self._commanager.updateCReward(self._arm[0], self._arm[1], reward[1])
			self._commanager.switchArm(self._arm[0])
		else:
			self._cbandit[self._arm[0]].updateEst(reward[1], self._arm[1])


	def _updateRewardCBandit(self, reward, updateparent=False):
		"""
		If updateparent is True, update the parents of community in heirarchy too
		"""
		cids = [self._arm[1]]

		if updateparent:
			cids += self._hcommunity.getAncestors(self._arm[0], self._arm[1])

		for cid in cids:
			self._cbandit[self._arm[0]].updateEst(reward, cid)


	def getNode(self, count=1):
		self._nextArms()

		while self._arm is None:
			self._nextArms()
		
		candidates = set(list(self._hcommunity.getNodes(self._arm[0], self._arm[1])))
		candidates.difference_update(self._queried[0])

		return self._rbandit[self._arm[0]].nextNode(candidates)


class RoleManager(object):
	"""docstring for RoleManager"""
	def __init__(self, sample, queried):
		super(RoleManager, self).__init__()
		self._sample = sample
		self._queried = queried
		
		self._roles = [('degree', 'highest'), ('degree', 'lowest'), \
		 ('betweeness', 'highest'), ('betweeness', 'lowest'),\
		 ('core', 'highest'), ('core', 'lowest'),
		 ('random','')]
	
		self._cache = [{} for _ in self._sample]
		

	def getRoles(self):
		return self._roles


	def clearCache(self):
		self._cache = [{} for _ in self._sample]


	def getNode(self, layer, role, nodes=None, count=1):
		"""
		Get node satisfying role from layer

		If nodes is given, select only from that list
		"""
		s = self._sample[layer]
		r = self._roles[role]

		candidates = set(list(s.nodes()))

		if nodes is not None:
			candidates.intersection(nodes)

		s = self._sample[0].subgraph(candidates)
		
		# Restrict candidates to only unqueried nodes
		candidates.difference_update(self._queried[0])

		if len(candidates) == 0:
			return False

		# Get nodes and values
		if r[0] == 'degree':
			vals = nx.degree_centrality(s)
		elif r[0] == 'betweeness':
			vals = nx.betweenness_centrality(s, k=min(10, s.number_of_nodes()))
		elif r[0] == 'closeness':
			vals = nx.closeness_centrality(s)
		elif r[0] == 'clustering':
			vals = nx.clustering(s)
		elif r[0] == 'eig':
			vals = nx.eigenvector_centrality_numpy(s, max_iter=100, tol=1e-2)
		elif r[0] == 'core':
			vals = nx.core_number(s)
		elif r[0] == 'random':
			candidates = list(candidates)
			np.random.shuffle(candidates)
			return candidates[:count]

		# Filter to only nodes in canditate list
		candidates = {u:vals[u] for u in candidates}

		# Sort by the value
		candidates = sorted(candidates, key=candidates.get)

		# Return highest or lowest depending on role
		if r[1] == 'lowest':
			return candidates[:count]
		else:
			return candidates[-count:]
		

class CommunitHeirarchy(object):
	"""docstring for CommunitHeirarchy"""
	def __init__(self, sample, mode=0):
		super(CommunitHeirarchy, self).__init__()
		self._sample = sample
		self._mode = mode
		self._initializeCommunities()


	def getLayerCount(self):
		return len(self._sample)


	def _initializeCommunities(self):
		"""
		Intitialize communites for each of the layers
		"""
		self._ocom = {}
		self._dendrogram = []
		self._com_ids = {}

		for i in range(0, len(self._sample)):
			partition, _ = getApproxPartition(self._sample[i], single=False)
			
			if self._mode == 0:
				com = mycommunity.Community(self._sample[i], partition)
				com.generateDendrogram(stop_max=False)
				tree = com.communityTree()
				ids = com.communityIds()

				self._ocom.append(com)
				self._dendrogram.append(com.flattenDendrogram(tree=tree))
				self._com_ids.append({i:ids[i] for i in range(0, len(ids))})

			else:
				self._com_ids[i] = list(set(partition.values()))
				self._ocom[i] = {c:[] for c in self._com_ids[i]}
				
				for u in partition:
					self._ocom[i][partition[u]].append(u)


	def getCommunityIds(self, layer):
		"""
		Get the community ids
		"""
		return self._com_ids[layer]


	def getNodes(self, layer, cid):
		"""
		Get the nodes in layer, and communit id
		"""
		if self._mode == 0:
			return self._ocom[layer].nodesInCommunity(cid)
		else:
			return self._ocom[layer][cid]


	def getRootCommunity(self):
		"""
		Get the root nodes of layers community
		"""
		if self._mode == 0:
			return [c.getRoot() for c in self._ocom]
		else:
			return self._com_ids


	def getChildren(self, layer, cid):
		"""
		Get the children of cid in layer
		"""
		if self._mode == 0:
			return self._dendrogram[layer][cid]['children']
		else:
			return None


	def getParent(self, layer, cid):
		"""
		Get the parent of cin in layer
		"""
		if self._mode == 0:
			return self._dendrogram[layer][cid]['parent']
		else:
			return None


	def getSibling(self, layer, cid):
		"""
		Get the sibling of cid in layer
		"""
		if self._mode == 0:
			return self._dendrogram[layer][cid]['siblings']
		else:
			return None


	def checkRoot(self, layer, cid):
		"""
		Check in cid is root in dendrogram
		Returns True if root, otherwis false
		"""
		if self._mode == 0:
			return self._dendrogram[layer][cid] is not None
		else:
			return True


	def checkLeaf(self, layer, cid):
		"""
		Check if cid is a leaf in dendrogram
		"""
		if self._mode == 0:
			return len(self._dendrogram[layer][cid]) == 0
		else:
			return True


	def getAncestors(self, layer, cid):
		"""
		Get all the ancestors of cid in layer
		"""
		if self._mode == 0:
			parent = self.getParent(layer, cid)

			if parent is None:
				return []

			return [parent] + self.getAncestors(layer, parent)
		else:
			return None

	def getDecendents(self, layer, cid):
		"""
		Get all decendents of cid in layer
		"""
		if self._mode == 0:
			children = self.getChildren(layer, cid)

			if len(children) == 0:
				return []

			return [ children[0], children[1] ] + self.getDecendents(layer, children[0]) + self.getDecendents(layer, children[1])
		else:
			return None
		

class MABSample(object):
	"""docstring for MABSample"""
	def __init__(self, graph, sample, lweight, lcosts, queried, budget, results_continuous=True, partition=None, bmode=1):
		super(MABSample, self).__init__()
		self._sample = sample
		self._graph = graph
		self._lweight = lweight
		self._lcosts = lcosts
		self._queried = queried
		self._budget = budget
		self._results_continuous = results_continuous

		#print('Bandit Manager')
		self._bandit = BanditManager(self._graph, self._sample, self._queried, commode=1, bmode=bmode)

		#print('Evaluation')
		self._evaluation = Evaluation(self._graph, partition=partition)

		self._step = max(5,int(self._graph[0].number_of_nodes()/100))
		self._window_size = 10
		self._importance_threshold = 2						# Minimum importance of cheap layers to aggregate
		self._scores = []
		self._rtime = []
		self._ppart = [None, None] 									# The previous best partition used for community update distance computation
		self._psim = 0										# Similarity between previous 2 partitions
		#self._bandit.initializeRBandits()
		self._node_sequence = []							# Sequence of nodes queried
		self._nodes_count = self._graph[0].number_of_nodes()
		self._batch_size = 1
		self._prevsim = [0.0, 0.0, 0.0]									# For blackbox reward


	def _initializeSample(self):
		"""
		Add nodes and edges from 'valid' layers to sample of interest
		"""
		importances = self._lweight.getLayerOverlap()

		edges_add = set([])									# edges to add
		edges_sub = set([])									# edges to remove

		for i in range(1, len(self._sample)):
			nodes = set(list(self._sample[i].nodes()))
			nodes.difference_update(self._queried[0]) 		# nodes that have not been queried in layer 0

			sg = self._sample[i].subgraph(nodes)

			edges = [frozenset(e) for e in sg.edges() if e[0] != e[1]]

			if importances[i] > self._importance_threshold:
				edges_add.update(edges)
			else:
				edges_sub.update(edges)

			# Edges to remove cannot be in edges to add
			edges_sub.difference_update(edges_add)

			# nodes always exist in 0 since its multiplex
			self._sample[0].add_nodes_from(nodes)

		# Update sample 0
		self._sample[0].add_edges_from([list(e) for e in edges_add])
		self._sample[0].remove_edges_from([list(e) for e in edges_sub])

		self._ppart[1], self._pmod = getApproxPartition(self._sample[0])


	def _getStreamingPartition(self, neighbors_all, subgraph):
		"""
		Compute partition after node u, with neighbors n
		is added to current sample
		sample is the current sample
		"""
		# Assign u to partition with most common neighbors
		# This gives max modularity
		"""
		Compute partition after node u, with neighbors n
		is added to current sample
		sample is the current sample
		"""
		# Assign u to partition with most common neighbors
		# This gives max modularity
		if subgraph.number_of_edges() == 0:
			cpart = {i:i for i in subgraph.nodes()}
			cmod = 1 				# So that next iteration there is proper community detection
			return cpart, cmod

		if len(self._queried[0]) < 10 or len(self._queried[0]) % 50 == 0:
			cpart = community.best_partition(subgraph, randomize=False)
			cmod = community.modularity(cpart, subgraph)
			return cpart, cmod

		tpart = dict(self._ppart[1])
		for u in neighbors_all:
			n = neighbors_all[u]

			candidates = self._queried[0].intersection(n)

			#for i in u:
			counts = {c:0 for c in set(self._ppart[1].values())}
			for v in candidates:
				if v in self._ppart[1]:
					counts[self._ppart[1][v]] += 1
			
			c = max(counts.items(), key=operator.itemgetter(1))[0]
			tpart[u] = c
		
		tmod = community.modularity(tpart, subgraph)

		if tmod < self._pmod:
			cpart, cmod = getApproxPartition(subgraph)
			return cpart, cmod

		else:
			return tpart, tmod 


	def _communityUpdateDistance(self, cpart, u):
		"""
		Compute the change in community between communities of the
		current sample and previous one
		"""
		nodes = self._queried[0].intersection(self._ppart[1].keys()).intersection(cpart.keys())
		
		part1 = [self._ppart[1][u] for u in nodes]
		part2 = [cpart[u] for u in nodes]
		
		nmi = 1 - normalized_mutual_info_score(part1, part2)

		if self._ppart[0] is None:
			return nmi

		# Find direction
		part1 = [self._ppart[0][u] for u in nodes if u in self._ppart[0]]
		part2 = [cpart[u] for u in nodes if u in self._ppart[0]]

		dnmi = 1 - normalized_mutual_info_score(part1, part2)
		direction = 1.0

		if dnmi < self._psim:
			direction = -1.0

		self._psim = dnmi

		return direction * dnmi


	def _rewards(self, cpart, u):
		dist = self._communityUpdateDistance(cpart, u)
		self._past_distances.append(np.abs(dist))

		return (dist, dist, dist)


	def _checkTerminate(self):
		"""
		Check to see if we should end current iteration of MAB
		"""
		if len(self._past_distances) < 5:
			return 0
		if np.mean(self._past_distances[-5:]) < 0.10:
			return 1
		if np.mean(self._past_distances[-5:]) < 0.10:
			return 2
		return 0


	def getScores(self):
		return self._scores


	def getRunningTime(self):
		return self._rtime


	def getNodeSequence(self):
		return self._node_sequence


	def _resultsStep(self):
		if self._graph[0].number_of_nodes() < 1000:
			return True
		step = self._step
		return self._budget.getBudgetConsumed() % step < 1


	def _blackBoxReward(self, cpart, u):
		sim = self._evaluation.communitySimilarity(self._sample[0], self._queried[0])
		
		j = 0
		if sim[2] < sim[1]:
			j = 2
		else:
			j = 1

		if self._prevsim[j] > 0:
			reward = (sim[j] - self._prevsim[j])/self._prevsim[j]
		else:
			reward = sim[j]

		self._prevsim = sim
		self._past_distances.append(np.abs(reward))

		return (reward, reward, reward)


	def _reorderNodes(self):
		tgraph = nx.Graph()
		
		edges = list(self._sample[0].edges())
		nodes = list(self._sample[0].nodes())
		nodes = sorted(nodes)
		edges = sorted(edges)

		tgraph.add_nodes_from(nodes)
		tgraph.add_edges_from(edges)

		self._sample[0] = tgraph

	def sample(self, budget):
		"""
		If budget is none, sample ends after distance is too small
		icost is the initial cost
		mcost is the maximum total cost allowed
		"""
		self._initializeSample()
		self._bandit.initializeCBandits()

		cost = 0
		mmode, tmmode = 0, 0		# mmode = 0: normal mab sampling; mmode = 1: MOD to stregththen current structure

		self._past_distances = []

		if self._results_continuous:
			self._reorderNodes()
			self._scores.append([self._budget.getBudgetConsumed()/self._nodes_count] +  self._evaluation.communitySimilarity(self._sample[0], self._queried[0]) + [len(self._queried[0]),self._sample[0].number_of_nodes(), self._sample[0].number_of_edges()])
			print('SCORE \t\t\t', self._scores[-1], mmode)
		
		while self._budget.getBudgetLeft() > 0:
			# Get next node
			if mmode == 0:
				u = self._bandit.getNode(self._batch_size)
				# No node was found so go to mmode 1 and penanlize current arms
				if u is False :
					self._bandit.updateReward([0,0,0], self._sample)
					tmmode = 1
					
			if mmode == 1 or tmmode == 1:
				deg = {v:len(set(list(self._sample[0][v]))) for v in self._sample[0].nodes() if v not in self._queried[0]}
				if len(deg) > 0:
					u = sorted(deg, key=deg.get, reverse=True)
					u = [u[0]]
				else:
					continue

				# Reset tmmode
				tmmode = 0

			# Query in layer 0
			neighbors_all, edges = {}, []
			for w in u:
				try:
					neighbors = self._graph[0].neighbors(w)
					edges += [(w,v) for v in neighbors]
				except:
					self._bandit.updateReward([-1.0, -1.0, -1.0], self._sample)
					break

			# Remove existing edges with u
			for w in u:
				try:
					self._sample[0].remove_node(w)
				except:
					pass
			self._sample[0].add_nodes_from(u)

			# Add the new edges
			self._sample[0].add_edges_from(edges)

			for w in u:
				neighbors = self._sample[0].neighbors(w)
				neighbors_all[w] = list(neighbors)

			# Update queried and cost
			cost += len(u) * self._lcosts[0]
			self._budget.consumeBudget(len(u)*self._lcosts[0])

			self._queried[0].update(u)
			self._node_sequence += u
		
			cpart, cmod = getApproxPartition(self._sample[0])

			# Reward for bandit and update
			reward = self._rewards(cpart, u)
			
			if mmode == 0:
				self._bandit.updateReward(reward, self._sample)

			self._lweight.updateLayerOverlap(self._sample, self._queried[0])

			if self._results_continuous and self._resultsStep():
				self._reorderNodes()
				b = self._budget.getBudgetConsumed()/self._nodes_count
				self._scores.append([b] + self._evaluation.communitySimilarity(self._sample[0], self._queried[0]) + [len(self._queried[0]), self._sample[0].number_of_nodes(), self._sample[0].number_of_edges()])
				self._rtime.append([b, 1000 * time.process_time()])
				print('SCORE \t\t\t', self._scores[-1], mmode)

			if cost > budget:
				t = self._checkTerminate()
				
				if t == 1:
					break
				elif t == 2:
					mmode = 1
				elif t == 0:
					mmode = 0

			self._ppart[0] = self._ppart[1]
			self._ppart[1] = cpart
			self._pmod = cmod

		#if not self._results_continuous or True:
		if self._budget.getBudgetConsumed()/self._nodes_count != self._scores[-1][0]:
			self._reorderNodes()
			b = self._budget.getBudgetConsumed()/self._nodes_count
			self._scores.append([b] + self._evaluation.communitySimilarity(self._sample[0], self._queried[0]) + [len(self._queried[0]),self._sample[0].number_of_nodes(), self._sample[0].number_of_edges()])
			self._rtime.append([b, 1000 * time.process_time()])
			print('SCORE \t\t\t', self._scores[-1], mmode)

	def getLayerEstimates(self):
		return self._bandit.getLayerEstimates()
		

class MultiPlexSampling(object):
	"""docstring for MultiPlexSampling"""
	def __init__(self, fnames=[], budget=0, costs=[], partition=None, bmode=1):
		super(MultiPlexSampling, self).__init__()
		self._graph = self._getMultiGraph(fnames, connected=False)
		self._sample = [nx.Graph() for _ in self._graph]
		self._partition = self._getPartition(partition)

		if budget == 0:
			budget = int(0.5 * self._graph[0].number_of_nodes())

		self._lcosts = costs[:len(self._graph)]
		self._queried = [set([]) for _ in self._graph]

		budget = int(budget * len(set([u for g in self._graph for u in g.nodes()])) )
		self._lweight = LayerImportance(len(self._graph))
		self._budget = Budget(budget, self._lcosts, self._lweight)
		self._roles = RoleManager(self._sample, self._queried)
		self._evaluation = Evaluation(self._graph, partition=self._partition)

		print('Initializing RND')
		self._rnd = RNDSample(self._graph, self._sample, self._lcosts, self._queried, self._budget, self._lweight)

		print('Initializing MAB')
		self._mab = MABSample(self._graph, self._sample, self._lweight, self._lcosts, self._queried, self._budget, partition=self._partition, bmode=bmode)


	def _getPartition(self, partition):
		# Check if file exist
		try:
			f = open(partition, 'rb')
			part = pickle.load(f)
			f.close()
		except Exception as e:
			f = open(partition, 'wb+')
			part, mod = getApproxPartition(self._graph[0])
			print('Communtiy Count: {} \t {}'.format(len(set(part.values())), mod))
			pickle.dump(part, f)
			f.close()	
		return part

	def _nodeAlign(self, graph, mode='sub'):
		"""
		Node align graph by making sure all layers have same nodes
		"""
		if mode == 'sub':
			nodes = set(graph[0].nodes())

			for g in graph[1:]:
				nodes = nodes.intersection(g.nodes())

			for g in graph:
				remove = set(g.nodes()).difference(nodes)
				g.remove_nodes_from(remove)
		elif mode == 'add':
			nodes = set([])
			for g in graph[1:]:
				nodes = nodes.union(g.nodes())

			for g in graph:
				add = nodes.difference(g.nodes())
				g.add_nodes_from(add)

		return graph


	def _getMultiGraph(self, fname, connected=True):
		"""
		Get the differet layers of graph

		file format: layerid node0 node1
		"""
		nodes, edges = set([]), {}
		with open(fname, 'r') as f:
			next(f)
			reader = csv.reader(f, delimiter='\t')
			for row in reader:
				l = int(row[0])
				u = row[1]
				v = row[2]
				nodes.update([u,v])
				if l not in edges:
					edges[l] = []
				if u != v:
					edges[l].append((u,v))

		mgraph = [None for _ in edges]
		for l in edges:
			g = nx.Graph()
			g.add_nodes_from(nodes)
			g.add_edges_from(edges[l])
			mgraph[l] = g

		tgraph = nx.Graph()
		edges = list(mgraph[0].edges())
		nodes = list(mgraph[0].nodes())
		nodes = sorted(nodes)
		edges = sorted(edges)

		tgraph.add_nodes_from(nodes)
		tgraph.add_edges_from(edges)

		mgraph[0] = tgraph

		return mgraph

	
	def _getGraph(self, fname, connected=True):
		"""
		Returns the graph
		If conncted is TRUE, returs the largest conncted component
		"""
		graph = nx.read_edgelist(fname, delimiter=',')
		if graph.number_of_nodes() == 0:
			graph = nx.read_edgelist(fname, delimiter=' ')
		if graph.number_of_nodes() == 0:
			graph = nx.read_edgelist(fname, delimiter='\t')

		# Remove self loops
		sedges = graph.selfloop_edges()
		graph.remove_edges_from(sedges)
		if connected:
			graph = self._getLargestComponent(graph)

		return graph


	def _getLargestComponent(self, graph):
		"""
		Returns the largest conncted component
		"""
		return max(nx.connected_component_subgraphs(graph), key=len)


	def run(self, _):
		random.seed()
		np.random.seed()

		budget = self._budget.initializeBudget()

		while self._budget.getBudgetLeft() > 0:
			print('Iteration RND')
			self._rnd.sample(budget)

			print('Iteration MAB')
			self._mab.sample(budget[0])

			self._budget.updateSlices()
			budget = self._budget.allocateBudget()
			
		return (self._mab.getScores(), self._mab.getNodeSequence(), self._mab.getRunningTime())


if __name__ == '__main__':
	sname = sys.argv[1]
	budget = float(sys.argv[2])
	exp = int(sys.argv[3])
	partition = sys.argv[4]
	dname = sys.argv[5]
	bmode = int(sys.argv[6])
	fnames = sys.argv[7]

	costs = [1] + [0.5]*25 # Query costs for the different layers
	#bmode = 1
	bnames = [None, 'Epsilon-Greedy', 'Epsilon-Decreasing', 'VDBE', 'UCB']

	aname = 'MCS+'

	aggregate_score = {}
	aggregate_time = {}

	mcs = MultiPlexSampling(fnames, budget, costs, partition, bmode)
	start_time = time.process_time()
	
	with Pool(multiprocessing.cpu_count()) as p:
		results = p.map(mcs.run, range(exp))

	for i in range(0, len(results)):
		score = results[i][0]
		rtime = results[i][2]

		for x in score:
			b = np.round(x[0], 2)
			if b not in aggregate_score:
				aggregate_score[b] = []
			aggregate_score[b].append(x[1])

		for x in rtime:
			b = np.round(x[0], 2)
			if b not in aggregate_time:
				aggregate_time[b] = []
			aggregate_time[b].append(x[1])

	
	with open(sname + '_aggregate.csv', 'a') as f:
		writer = csv.writer(f, delimiter='\t')
		sc = sorted(aggregate_score.keys(), reverse=False)
		for b in sc:
			scores = aggregate_score[b]

			m = np.mean(scores)
			s = np.std(scores)

			print('{}\t{}\t{}'.format(b,m,s))
			writer.writerow([b, m, s, dname, bnames[bmode], aname])
		f.close()

	with open(sname + '_time.csv', 'a') as f:
		writer = csv.writer(f, delimiter='\t')
		sc = sorted(aggregate_time.keys(), reverse=False)
		for b in sc:
			scores = [t - start_time for t in aggregate_time[b]]

			m = np.mean(scores)
			s = np.std(scores)
			writer.writerow([b, m, s, dname, bnames[bmode], aname])
		f.close()


	



