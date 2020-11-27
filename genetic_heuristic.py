import numpy as np 
import json
import time
import itertools
import random
from copy import deepcopy
import threading
import sys
import os


dict_data = json.load(open("config.json","r"))
start=time.time()
items = list(dict_data['profits'].keys())
n_item = len(items)
profits = dict_data['profits']
costs = dict_data['costs']
W = dict_data['knapsack_size']
synergies = dict_data['synergies']
Gamma = min(dict_data['Gamma'], n_item)
population = []
parents = []
siblings = []
terminating = []
reinitialize = False
optimum = []
bestSol = (0,0)

#THREAD INITIALISING ONE SOLUTION
class initialisingThread(threading.Thread):
	def __init__(self, item):
		global profits, costs, synergies
		threading.Thread.__init__(self)
		self.item = item
		self.sol = [self.item]
		self.best = []

	def run(self):
		global population, items, profits, costs, synergies
		self.recursiveBuilding(self.sol, False)
		self.best = deepcopy(self.sol)	
		self.nominalItems = list(set(items) - set(self.best))
		if reinitialize:
			self.nominalItems.sort(key = lambda i: profits[i] - costs[i]['a'] +\
			 np.sum([synergies[i][j] for j in self.best]) , reverse = True)
		else:
			#DEVIATIONS
			self.nominalItems.sort(key = lambda x: costs[x]['u'] - costs[x]['a'], reverse = True)

		for j in self.nominalItems:
			test = self.best +[j]
			if self.isFeasible(test):
				self.best = deepcopy(test)
		population.append(self.best)

	def recursiveBuilding(self, sol, stop):
		global Gamma, profits, costs, synergies

		if len(self.sol) == Gamma :
			self.best = deepcopy(self.sol)
			stop = True
			return
		if not stop:
			remaining = list(set(items) - set(self.sol))

			if reinitialize:
				remaining.sort(key = lambda x: profits[x] - costs[x]['u'] +\
					np.sum([synergies[x][j] for j in self.sol]), reverse = True)
			else:
				#DEVIATIONS
				remaining.sort(key = lambda x: costs[x]['u'] - costs[x]['a'], reverse = True)
			for j in remaining:
				test = self.sol + [j]
				if self.isFeasible(test):
					self.sol = deepcopy(test)
					self.recursiveBuilding(self.sol, False)
				else:
					test = deepcopy(self.sol)

	def isFeasible(self,solution):
		global Gamma, W			

		if len(solution) != len(set(solution)): #double Elements
			return False
		if len(solution) <= Gamma and self.evaluateCost(solution) <= W:
			return True
		elif len(solution) > Gamma and self.evaluateCost(solution) <= W:
			deviationNominal = [costs[i]['u'] - costs[i]['a'] for i in solution[Gamma:]]
			deviationUpper = [costs[i]['u'] - costs[i]['a'] for i in solution[:Gamma]]
			return all(dn <= du for dn in deviationNominal for du in deviationUpper)
		else:
			return False

	def evaluateCost(self,solution):
		global Gamma, costs, W
		if len(solution) <= Gamma:
			return np.sum([costs[i]['u'] for i in solution])
		else:
			return np.sum([costs[i]['u'] for i in solution[:Gamma]]) + np.sum([costs[i]['a'] for i in solution[Gamma:]])

#THREAD CROSSOVER
class crossoverThread(threading.Thread):
	def __init__(self, parents, threadID):
		global terminating, siblings
		threading.Thread.__init__(self)
		self.parents = parents
		self.threadID = threadID
		self.siblings = []

	def run(self):
		global population, terminating, reinitialize

		if len(self.parents) == 2:
			self.classicalParenting()
		for son in self.siblings:
			siblings.append(son)
		terminating[self.threadID] = True

	def classicalParenting(self):
		point = random.randint(0, np.min([len(self.parents[0])-1, len(self.parents[1])-1]))
		self.siblings +=[
			self.parents[0][:point] + self.parents[1][point:],
			self.parents[1][:point] + self.parents[0][point:]
		]

	def isFeasible(self,solution):
		global costs, Gamma, W	

		if len(solution) <= Gamma:
			costOfSolution = np.sum([costs[i]['u'] for i in solution])
		else:
			costOfSolution = np.sum([costs[i]['u'] for i in solution[:Gamma]]) + np.sum([costs[i]['a'] for i in solution[Gamma:]])
		if len(solution) != len(set(solution)): #double Elements
			return False
		if len(solution) <= Gamma and costOfSolution <= W:
			return True
		elif len(solution) > Gamma and costOfSolution <= W:
			deviationNominal = [costs[i]['u'] - costs[i]['a'] for i in solution[Gamma:]]
			deviationUpper = [costs[i]['u'] - costs[i]['a'] for i in solution[:Gamma]]
			return all(dn <= du for dn in deviationNominal for du in deviationUpper)
		else:
			return False

class mutationThread(threading.Thread):
	def __init__(self, threadID, elem):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.elem = elem

	def run(self):
		global terminating
		self.mutation()
		terminating[self.threadID] = True

	def mutation(self):
		global items, profits, costs, synergies, Gamma, population, W
		missingElems = list(set(items) - set(self.elem))
		if missingElems:
			pos = random.randint(0, len(self.elem)-1)
			if pos < Gamma:
				missingElems.sort(key = lambda x: profits[x] - costs[x]['u'] + np.sum([synergies[x][j] for j in self.elem if j!=self.elem[pos]]), reverse = True)
			else:
				missingElems.sort(key = lambda x: profits[x] - costs[x]['a'] + np.sum([synergies[x][j] for j in self.elem if j!=self.elem[pos]]), reverse = True)
			self.elem[pos] = missingElems.pop(0)
		else:
			pos = random.randint(0, len(self.elem)-1)
			self.elem.pop(pos)	
		while not self.isFeasible(self.elem):
			self.elem.pop(-1)
		population.append(self.elem)

	def isFeasible(self,solution):
		global costs, Gamma, W	

		if len(solution) <= Gamma:
			costOfSolution = np.sum([costs[i]['u'] for i in solution])
		else:
			costOfSolution = np.sum([costs[i]['u'] for i in solution[:Gamma]]) + np.sum([costs[i]['a'] for i in solution[Gamma:]])
		if len(solution) != len(set(solution)): #double Elements
			return False
		if len(solution) <= Gamma and costOfSolution <= W:
			return True
		elif len(solution) > Gamma and costOfSolution <= W:
			deviationNominal = [costs[i]['u'] - costs[i]['a'] for i in solution[Gamma:]]
			deviationUpper = [costs[i]['u'] - costs[i]['a'] for i in solution[:Gamma]]
			return all(dn <= du for dn in deviationNominal for du in deviationUpper)

		else:
			return False

def createPopulation():
	global n_item, items
	initialLength = len(population)
	for item in items:
		t = initialisingThread(item)
		t.start()
	
	while len(population)!=initialLength + n_item:
		continue	

def evaluateProfit(solution):
	global profits
	return np.sum([profits[i] for i in solution])

def evaluateCost(solution):
	global Gamma, costs, W
	if len(solution) <= Gamma:
		return np.sum([costs[i]['u'] for i in solution])
	else:
		return np.sum([costs[i]['u'] for i in solution[:Gamma]]) + np.sum([costs[i]['a'] for i in solution[Gamma:]])

def evaluateSynergies(solution):
	global synergies
	indexes = list(itertools.combinations(solution, 2))
	return np.sum([synergies[tup[0]][tup[1]] for tup in indexes])

def fitnessScore(solution):
	return evaluateProfit(solution) - evaluateCost(solution) + evaluateSynergies(solution)
	
def objFunct(solution):
	return evaluateProfit(solution) - evaluateCost(solution) + evaluateSynergies(solution)

def parentsSelection():
	global population, n_item, Gamma, siblings
	population.sort()
	population = list(k for k,_ in itertools.groupby(population))
	population.sort(key=lambda elem: fitnessScore(elem), reverse = True)
	population = deepcopy(population[:int(n_item)])
	siblings = []


def crossoverProcedure(it):
	global population, terminating, parents

	doubleParents = list(itertools.combinations(population[:int(len(population)/it)],2))
	parents = doubleParents
	crossOvers = []
	for i in range(len(parents)):
		terminating.append(False)
		c = crossoverThread(parents[i], i)
		crossOvers.append(c)	
	for c in crossOvers:
		c.start()
	while sum(terminating)!=len(terminating):
		continue

	terminating = []

def mutationProcedure(prob):
	global population, terminating

	mutations = []
	j = 0
	for i in range(len(siblings)):
		drawn = random.uniform(0,1)
		if drawn <= prob:
			terminating.append(False)
			m = mutationThread(j, siblings[i])
			mutations.append(m)
			j+=1
	for m in mutations:
		m.start()
	while sum(terminating)!=len(terminating):
		continue
	terminating = []

def getOptimum():
	global population, optimum, bestSol
	#print("Optimum Solution {}, Value: {}".format(population[0], round(objFunct(population[0]),2)))
	population.sort(key=lambda elem: fitnessScore(elem), reverse = True)
	optimum.append(objFunct(population[0]))
	if bestSol[1] < objFunct(population[0]):
		bestSol = (population[0], objFunct(population[0]))

def isEvolving():
	global optimum, Gamma
	if abs(optimum[-5] - optimum[-1]) < 0.0001 * optimum[-5]:
		return False
	return True

def main():
	global n_item, Gamma, reinitialize, population
	createPopulation()
	#print(population)
	for i in range(0,n_item):
		random.seed(i)
		if i > 5:
			if not isEvolving():
				reinitialize = True
				break	
		parentsSelection()
		crossoverProcedure(1+(i)/5)
		mutationProcedure(min(0.2*(i+1),1))
		getOptimum()
	if reinitialize:
		population = []
		createPopulation()
		print(population)
		for j in range(i,n_item):
			#random.seed(random.randint(0,n_item))
			random.seed(j)
			parentsSelection()
			crossoverProcedure(1+(j-i+1)/5)
			mutationProcedure(1)
			getOptimum()
			if not isEvolving():
				break
	
	end = time.time()
	print("Time from the start: {}".format(end - start))
	print("Best Solution: {}, {}".format(bestSol[0],bestSol[1]))
	
main()



