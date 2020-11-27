import json
import time
import itertools
import numpy as np
import random
import os

dict_data = json.load(open("config.json","r"))

class Node(object):
    def __init__(self, key):
        self.key = key
        if (key != "source" and key != "destination"):
            self.costUpper = dict_data['costs'][key]['u']
            self.costNominal = dict_data['costs'][key]['a']
            self.profit = dict_data['profits'][key]
        else:
            self.costUpper = 0
            self.costNominal = 0
            self.profit = 0

    def getKey(self):
        return self.key

    def getCostUpper(self):
        return self.costUpper

    def getCostNominal(self):
        return self.costNominal

    def getDeviation(self):
        return self.costUpper - self.costNominal

    def getProfit(self):
        return self.profit

class LightArc(object):
    def __init__(self, sourceNode, destinationNode):
        self.source = sourceNode
        self.destination = destinationNode

    def getSource(self):
        return self.source

    def getDestination(self):
        return self.destination

class HeavyArc(object):
    def __init__(self, sourceNode, destinationNode):
        self.source = sourceNode
        self.destination = destinationNode

    def getSource(self):
        return self.source

    def getDestination(self):
        return self.destination

class Graph(object):
    def __init__(self, graph, nodes, lightEdges, heavyEdges):
        self.nodes = nodes
        self.lightEdges = lightEdges
        self.heavyEdges = heavyEdges
        self.graph = graph

        #print("\nGraph Created with {} Vertices and {} Edges".format(str(len(self.nodes)),str(len(self.lightEdges)+len(self.heavyEdges))))

    def getNodes(self):
        return self.nodes

    def getArcs(self):
        return self.edges

    def getNeighbours(self, node):
        neighbours = []
        for edge in self.graph[node.getKey()]["heavy"]:
            neighbours.append(edge.getDestination())

        return neighbours

    def getPath(self):
        self.bestSol = 0
        self.bestPath = []
        

        for s in self.nodes:
            #first node is always "source"
            startNode = self.nodes[0] #fake "source" node
            partialSol = [startNode]

            #now pick each time a different starting node
            if s.getKey() !="source":
                #print("\nStarting procedure with node {}".format(s.getKey()))
                partialSol.append(s)

                start1 = time.time()
                self.greedy_path(s, partialSol)
                end1 = time.time()
                
                #print("Running time for the last iteration: {}".format(str(end1-start1)))
                #print("Best path: ", self.bestPath)
                #print("Current solution: {}".format(str(self.bestSol)))

        print("\n\n***FINAL RESULT***\nOptimal path: "+str(self.bestPath))
        print("Optimal solution: {}".format(str(self.bestSol)))


    def greedy_path(self,start_node,partialSol):

        #select greedily at each iteration the most convenient investment to pick 
        proceed=True
        
        #save inside here the last solution proposed to Build_Destroy: if it is proposed twice consecutively, stop
        currentCost = self.evaluateCost(partialSol)
        currentProfit = self.evaluateProfit(partialSol)

        currentSol = currentProfit - currentCost
        current_sol_syn = currentSol + self.evaluateSynergies(partialSol)

        new_path = [elem.getKey() for elem in partialSol]
        
        while proceed==True:
            
            #flag: if it remains False until the end of the iteration, exit this loop (no more arcs can be added)
            new_modification=False

            #arrange all the arcs using the lastly included node as new starting node
            heavyArcs = self.graph[partialSol[-1].getKey()]['heavy']
            lightArcs = self.graph[partialSol[-1].getKey()]['light']

            heavyArcs.sort(
            key= lambda edge: edge.getDestination().getProfit() - edge.getDestination().getCostUpper() +\
             np.sum([dict_data['synergies'][elem.getKey()][edge.getDestination().getKey()]\
             for elem in partialSol[1:] if elem != edge.getDestination()]),reverse = True)
            
            lightArcs.sort(
            key= lambda edge: edge.getDestination().getProfit() - edge.getDestination().getCostNominal() +\
             np.sum([dict_data['synergies'][elem.getKey()][edge.getDestination().getKey()]\
             for elem in partialSol[1:] if elem != edge.getDestination()]),reverse = True)
            
            if len(partialSol)-1<dict_data['Gamma']:
                #not gamma upper-cost items inserted: try to insert one
                #try to pick the best heavy arc
                ok=False
                cnt=0
                while ok==False and cnt<(dict_data["n_items"]-1):
                    edge=heavyArcs[cnt]
                    cnt+=1
                    
                    if (edge.getDestination() not in partialSol) and (currentCost + edge.getDestination().getCostUpper() <= dict_data['knapsack_size']):
                        #heavy arc fits in the capacity and does not go towards an already inserted node
                        #print("Adding edge towards "+str(edge.getDestination().getKey())+" with weight "+str(edge.getWeight()))
                        ok=True
                        dest = edge.getDestination()
                        partialSol.append(dest)
                        currentCost = self.evaluateCost(partialSol)
                        currentProfit = self.evaluateProfit(partialSol)
        
                        currentSol = currentProfit - currentCost
                        current_sol_syn = currentSol + self.evaluateSynergies(partialSol)

                        #if new solution is an improvement, update
                        if current_sol_syn > self.bestSol:
                            path = [elem.getKey() for elem in partialSol]                    
                            self.bestSol = current_sol_syn
                            self.bestPath = path
                        
                        new_modification=True

            else:
                #insert light arc, as gamma upper-cost elements are already present
                # insert best nominal and then check feasibility with build_Destroy
                for edge in lightArcs:

                    currentCost = self.evaluateCost(partialSol)

                    if edge.getDestination() not in partialSol  and currentCost + edge.getDestination().getCostNominal() <= dict_data['knapsack_size']:

                        dest = edge.getDestination()
                        partialSol.append(dest)
                        partialSolold=partialSol.copy()

                        current_sol_syn = self.buildDestroy(partialSol)

                        if len(partialSol)==len(partialSolold):
                            #solution was feasible, so let's keep it with a new nominal item
                            #check if we have a new best solution
                            if current_sol_syn > self.bestSol:
                                path = [elem.getKey() for elem in partialSol]                    
                                self.bestSol = current_sol_syn
                                self.bestPath = path

                            new_modification=True
                            break
                        else:
                            partialSol=partialSolold[:-1]
                        

            #an arc (if possible) has been added: now evaluate the new solution or exit if no new modification has been made
            if new_modification==False:
                #if finished with this source node: exit and go at next iteration of getPath()
                proceed=False 
        return
        
        
    
    def evaluateCost(self,partialSol):

        if (len(partialSol)-1 > dict_data['Gamma']):
            costs = [elem.getCostUpper() for elem in partialSol[1:dict_data['Gamma']+1]] + [elem.getCostNominal() for elem in partialSol[dict_data['Gamma']+1:]]

        else:
            costs = [elem.getCostUpper() for elem in partialSol]

        return np.sum(costs)

    def evaluateProfit(self,partialSol):

        profits = [elem.getProfit() for elem in partialSol]
        return np.sum(profits)

    def evaluateSynergies(self, partialSol):
        path = [elem.getKey() for elem in partialSol]

        synergiesProfit = 0

        synergies = list(itertools.combinations(path[1:],2))
        for tup in synergies:
            synergiesProfit += dict_data['synergies'][tup[0]][tup[1]]

        return synergiesProfit

    def isFeasible(self, partialSol):
        
        if len(partialSol) -1 > dict_data['Gamma']:
            variatedElements = partialSol[1:dict_data['Gamma']+1]
            nominalElements = partialSol[dict_data['Gamma']+1:]

            deviationsVariatedElements = [elem.getDeviation() for elem in variatedElements]
            deviationsNominalElements = [elem.getDeviation() for elem in nominalElements]

            for dev in deviationsNominalElements:
                if dev > np.min(deviationsVariatedElements):
                    return False

        return True

    def buildDestroy(self, partialSol):

        if self.isFeasible(partialSol):
            return self.evaluateProfit(partialSol) - self.evaluateCost(partialSol) + self.evaluateSynergies(partialSol)

        else:
            while not self.isFeasible(partialSol):

                variatedElements = partialSol[1:dict_data['Gamma']+1]
                nominalElements = partialSol[dict_data['Gamma']+1:]

                deviationsVariatedElements = [elem.getDeviation() for elem in variatedElements]
                deviationsNominalElements = [elem.getDeviation() for elem in nominalElements]

                minVariation = np.min(deviationsVariatedElements)
                for dev in deviationsNominalElements:
                    if dev > minVariation:
                        path = [elem.getKey() for elem in partialSol]
                        indexNominal = deviationsNominalElements.index(dev)
                        indexDeviated = deviationsVariatedElements.index(minVariation)

                        swap = partialSol[indexDeviated+1]
                        partialSol[indexDeviated+1] = partialSol[dict_data['Gamma']+1+indexNominal]
                        partialSol[dict_data['Gamma']+1+indexNominal] = swap
                        break
                                
                while self.evaluateCost(partialSol) > dict_data['knapsack_size']:
                    diffCost = self.evaluateCost(partialSol) - dict_data['knapsack_size']
                    nominalElements = [elem for elem in partialSol[dict_data['Gamma']+1:]]

                    nominalElements.sort(key = lambda elem: elem.getProfit() - elem.getCostNominal()\
                     + np.sum([dict_data['synergies'][elem.getKey()][node.getKey()] for node in partialSol[1:] if elem != node]), reverse = True)

                    if len(nominalElements)!=0:
                        elemToDelete = nominalElements[-1]
                    else:
                        upperElements = [elem for elem in partialSol[1:dict_data['Gamma']+1]]
                        upperElements.sort(key = lambda elem: elem.getProfit() - elem.getCostUpper()\
                         + np.sum([dict_data['synergies'][elem.getKey()][node.getKey()] for node in partialSol[1:] if elem != node]), reverse = True)
                        elemToDelete = upperElements[-1]
                    partialSol.remove(elemToDelete)

        return self.evaluateProfit(partialSol) - self.evaluateCost(partialSol) + self.evaluateSynergies(partialSol)

start = time.time()
graph = {}
source = Node("source")
nodes = [source]

graph["source"] = {
    "light":[],
    "heavy":[]
    }

for i in range(dict_data['n_items']):
    n = Node(str(i))
    nodes.append(n)
    graph[str(i)] = {
    "light":[],
    "heavy":[]
    }


lightEdges = []
heavyEdges = []

for node in nodes:
    for node2 in nodes:
        if node2 != node and node2.getKey()!= "source" and node.getKey() != "destination":
            l = LightArc(node, node2)
            h = HeavyArc(node, node2)
            lightEdges.append(l)
            heavyEdges.append(h)
            graph[node.getKey()]['light'].append(l)
            graph[node.getKey()]['heavy'].append(h)


g = Graph(graph, nodes, lightEdges, heavyEdges)

g.getPath()

end = time.time()
print("Running Time: {}".format(str(end-start)))



