import time
import logging
from pulp import *


class RobustKnapsack():
	def __init__(self):
		pass

	def solve(self, dict_data, time_limit=None, gap=None, verbose=False):

		logging.info("#########")
		items = range(dict_data['n_items'])

		x = LpVariable.dicts(
			"X", items,
			lowBound=0,
			cat='Binary'
		)
		
		y = LpVariable.dicts(
			"Y", (items, items),
			lowBound=0,
			cat='Binary')

		type_cost =['a','u','l']

		pi = LpVariable.dicts(
			"Pi",items,
			lowBound = 0,
			cat='Continuous')

		rho = LpVariable("Rho",lowBound=0, cat='Continuous')

		problem_name = "RobustKnapsack"

		prob = LpProblem(problem_name, LpMaximize)

		prob += lpSum([dict_data['profits'][str(i)]*x[i] for i in items]) + lpSum([dict_data['synergies'][str(i)][str(j)] * y[int(i)][int(j)]  for i in items for j in items if j>i]) - lpSum([dict_data['costs'][str(i)]['a'] * x[i]  for i in items]) - dict_data['Gamma'] * rho - lpSum([pi[i] for i in items]), "obj_func"

		prob += lpSum([dict_data['costs'][str(i)]['a'] * x[i]  for i in items]) + (dict_data['Gamma'] * rho) + lpSum([pi[i] for i in items]) <= dict_data['max_cost'], "budget"

		for i in items:
			prob += (rho + pi[i]) >= (dict_data['costs'][str(i)]['u'] - dict_data['costs'][str(i)]['a']) * x[i], "duality_constraint_Item_{}".format(str(i))

		for i in items:
			for j in items:
				if i!=j:
					prob += y[i][j] >= x[i] + x[j] - 1 , "synergies_AND1 {}".format(str(i)+"-"+str(j))
					prob += y[i][j] <= x[j] , "synergies_AND2 {}".format(str(i)+"-"+str(j))
					prob += y[i][j] <= x[i], "synergies_AND3 {}".format(str(i)+"-"+str(j))
		
		prob.writeLP("./logs/{}.lp".format(problem_name))

		msg_val = 1 if verbose else 0
		start = time.time()

		prob.solve(solver=COIN_CMD())
		end = time.time()

		#for i in items:
		#	print("Stock {0}, Value {1}".format(str(i),str(x[i].varValue)))
		logging.info("\t Status: {}".format(LpStatus[prob.status]))

		sol = prob.variables()
		of = value(prob.objective)
		valuesPi = [pi[i].varValue for i in items]
		#print("The value of the obj func: {}".format(str(of)))
		comp_time = end - start

		sol_x = [0] * dict_data['n_items']
		for var in sol:
			logging.info("{} {}".format(var.name, var.varValue))
			if "X_" in var.name:
				sol_x[int(var.name.replace("X_", ""))] = abs(var.varValue)
		logging.info("\n\tof: {}\n\tsol:\n{} \n\ttime:{}".format(
			of, sol_x, comp_time)
		)
		logging.info("#########")
		return of, sol_x, comp_time
