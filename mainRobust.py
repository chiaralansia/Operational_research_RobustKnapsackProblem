import json
import logging
import numpy as np
from Instance import Instance
from RobustKnapsack import RobustKnapsack
#from heuristic.simpleHeu import SimpleHeu


np.random.seed(0)


if __name__ == '__main__':
	log_name = "logs/main.log"
	logging.basicConfig(
		filename=log_name,
		format='%(asctime)s %(levelname)s: %(message)s',
		level=logging.INFO, datefmt="%H:%M:%S",
		filemode='w'
	)

	fp = open("config.json", 'r')
	sim_setting = json.load(fp)
	fp.close()

	inst = Instance(sim_setting)
	dict_data = inst.get_data()

	prb = RobustKnapsack()
	of_exact, sol_exact, comp_time_exact = prb.solve(
		dict_data,
		verbose=True
	)
	print(of_exact, sol_exact, comp_time_exact)

	