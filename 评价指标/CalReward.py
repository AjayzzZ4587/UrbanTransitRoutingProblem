import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import copy
import math

NUM_ROUTE = 4

# 站点数量
NUM_NODE = 15
MIN_NODE = 2

# 测试的代码类型
NETWORK_TYPE = "mandl1"

# 换乘概率
W_SHORTPATH= 0.05
W_TRANSFER= 1
W_CO= 1


# 个体类
class Individual:
    def __init__(self, chromosome, graph = nx.Graph()):
        self.chromosome = chromosome    #Exp: [0,0,0,0,0,0]
        self.fitness = 0                #Calculated Fitness
        self.graph = graph              # 接受一个图
        self.d0 = 0
        self.d1 = 0
        self.d2 = 0
        self.dun = 0
        self.travelTime = 0
        self.atravelTime = 0
        self.opcost = 0

demandMatrix = [[0 for j in range(NUM_NODE)] for i in range(NUM_NODE)]
pos = {}
def resetNetwork():
    demandMatrix = [[0 for j in range(NUM_NODE)] for i in range(NUM_NODE)]
    pos = {}

# 将数据引入图中
def createNetwork():
    G = nx.Graph()
    # Nodes
    f = open(
        f"C:/Users/45130/Desktop/最近在看的论文/POMO/NEW_py_ver/TNDP/input/{NETWORK_TYPE}_nodes.txt",
        "r")
    count = 0   # 站点数
    nodes = []  # 站点信息
    for line in f:
        if count > 0:
            nodes.append(line.split(","))
        count += 1
    f.close()

    for x in nodes:
        G.add_node(int(x[0]))
        pos[int(x[0])] = [float(x[1]), float(x[2])]
    # print(pos)
    # link
    f = open(
        f"C:/Users/45130/Desktop/最近在看的论文/POMO/NEW_py_ver/TNDP/input/{NETWORK_TYPE}_links.txt",
        "r")
    count = 0
    link = []
    for line in f:
        if count > 0:
            link.append(list(map(int, line.split(","))))
        count += 1
    f.close()
    for x in link:
        G.add_edge(x[0], x[1], weight=x[2])
    # demand
    f = open(
        f"C:/Users/45130/Desktop/最近在看的论文/POMO/NEW_py_ver/TNDP/input/{NETWORK_TYPE}_demand.txt",
        "r")
    count = 0
    demand = []
    for line in f:
        if count > 0:
            demand.append(list(map(int, line.split(","))))
        count += 1
    f.close()
    totalDemand = 0
    for line in demand:
        totalDemand += line[2]
        demandMatrix[line[0] - 1][line[1] - 1] = line[2]
    for x in demand:
        G.add_edge(x[0], x[1], demand=x[2])

    return G, totalDemand

#Initialize population
def initialize():
    population = []
    chromosome = []
    chromosome.append([1, 2, 3, 6, 8, 10, 11, 12])
    chromosome.append([2, 5, 4, 6, 8, 10, 13, 11])
    chromosome.append([9, 15, 7, 10, 8, 6, 4, 12])
    chromosome.append([4, 2, 3, 6, 15, 7, 10, 14])
    tempG = nx.Graph()
    for x in chromosome:
        nx.add_path(tempG, x)
    for x in tempG.edges():
        tempG.add_edge(x[0], x[1], weight=G[x[0]][x[1]]["weight"])
    population.append(Individual(chromosome, tempG))

    return population

def fitnessCalc(population):
    print(population)
    for index, x in enumerate(population):
        print(index)
        print(x)
        #print(x.chromosome)
        chromostr = []
        Co = 0
        #Cr = 0
        for i, y in enumerate(x.chromosome):
            routeCost = 0
            for j, z in enumerate(y):
                if j < len(y)-1:        # 假设还在这个范围内，其实也是因为最后一步不计算了
                    distance = x.graph[x.chromosome[i][j]][x.chromosome[i][j+1]]["weight"]
                    routeCost += distance
                    Co += distance
            #routeDemand = demandMatrix[x.chromosome[i][0]-1][x.chromosome[i][-1]-1]
            #Cr += routeCost*routeDemand
            chromostr.append(",".join(map(str, y)))
            y.reverse()
            chromostr.append(",".join(map(str, y)))
        #print(chromostr)
        # 记录站点到站点间的最短路径
        len_path = dict(nx.all_pairs_dijkstra(x.graph))
        # print(len_path[3][0][1]) source, len, target
        # print(len_path[3][1][1]) source, path, target
        d0 = 0
        d1 = 0
        d2 = 0
        dun = 0
        totalTravelTime = 0
        total = 0
        cost = 0
        cost2 = 0
        cost3 = 0
        #totalDemand = 0
        for i, y in enumerate(demandMatrix):
            for j, z in enumerate(y):
                # 记录最短走的那条路径是什么
                path = len_path[i+1][1][j+1]
                oripath = ",".join(map(str, path))
                # 记录从这走需要的时间
                travelTime = len_path[i+1][0][j+1]
                count = 0
                c0=0
                c1=0
                c2=0
                cun=0
                while path:
                    k = len(path)
                    while k >= 0:
                        pathstr = ",".join(map(str, path[:k]))
                        if any(pathstr in item for item in chromostr):
                            # print("Found: ", pathstr)
                            if k == len(path):
                                path = path[k:]
                                break
                            else:
                                path = path[k-1:]
                                count += 1
                                break
                        else:
                            k -= 1
                #print("Path String: ", oripath)
                if count == 0:
                    c0+=1
                    d0 += demandMatrix[i][j]
                elif count == 1:
                    c1+=1
                    d1 += demandMatrix[i][j]
                elif count == 2:
                    c2+=1
                    d2 += demandMatrix[i][j]
                else:
                    cun+=1
                    dun += demandMatrix[i][j]
                #print("Transfer: ", count)
                #print(f"Travel Time [{i+1}][{j+1}]: ", travelTime)
                travelTransfer = travelTime + count*5
                #print(f"Travel Time + Tansfer [{i+1}][{j+1}]: ", travelTransfer)
                totalTravelTime += travelTransfer
                cost += demandMatrix[i][j]*travelTransfer
                cost2 += demandMatrix[i][j]*travelTime
                cost3 += demandMatrix[i][j]*count
                total += 1
                #totalDemand += demandMatrix[i][j]
                # print(f"Path[{i+1}][{j+1}]: ", len_path[i+1][1][j+1])
                # print(f"Length[{i+1}][{j+1}]: ", len_path[i+1][0][j+1])
                # print("")
        #print("d0: ", d0)
        population[index].d0 = d0
        #print("d1: ", d1)
        population[index].d1 = d1
        #print("d2: ", d2)
        population[index].d2 = d2
        #print("dun: ", dun)
        population[index].dun = dun
        #print("Total travel Time: ", totalTravelTime)
        population[index].travelTime = cost
        att = cost/totalDemand
        population[index].atravelTime = att
        #print("Average Travel Time: ", att)
        #print("Total Cost: ", cost)
        #print("Operator Cost: ", Co)
        population[index].opcost = Co
        #print("Route Cost: ", Cr)
        #print("Total Demand: ", totalDemand)
        FitnessVal = cost2*W_SHORTPATH+cost3*W_TRANSFER+Co*W_CO
        #print("Objective Function", FitnessVal)
        population[index].fitness = FitnessVal
    return population




if __name__ == "__main__":
    G, totalDemand = createNetwork()
    population = initialize()

    y = fitnessCalc(population)

    lastChange = 0
    bestFitness = math.inf
    bestChromosome = []
    bestd0 = []
    bestd1 = []
    bestd2 = []
    bestdun = []
    bestTT = []
    bestATT = []
    bestNetwork = []
    bestGraph = nx.Graph()
    population = sorted(population, key=lambda x: x.fitness)
    fitness = np.array([(y.fitness) for y in population])

    bestFitness = population[0].fitness
    bestChromosome = population[0].chromosome
    bestd0 = (population[0].d0 / totalDemand) * 100
    bestd1 = (population[0].d1 / totalDemand) * 100
    bestd2 = (population[0].d2 / totalDemand) * 100
    bestdun = (population[0].dun / totalDemand) * 100
    bestTT = population[0].travelTime
    bestATT = population[0].atravelTime
    bestNetwork = population[0].opcost
    bestGraph = population[0].graph

    Reward = 2 * bestATT + 1.5 * (
                2 * (-bestd0 / 100) + 1.5 * bestd1 / 100 + 1 * bestd2 / 100) + 5 * bestdun + bestNetwork / 100 * 1.5
    print(Reward)
    print("Average Fitness: ", np.mean(fitness))
    print("---------------------------------------------")
    print("Best Chromosome: ", bestChromosome)
    print("Best Fitness: ", bestFitness)
    print("Best d0: %.2f" % bestd0)
    print("Best d1: %.2f" % bestd1)
    print("Best d2: %.2f" % bestd2)
    print("Best dun: %.2f" % bestdun)
    print("Best Travel Time: ", bestTT)
    print("Best Average Travel Time: %.2f" % bestATT)
    print("Network Cost:", bestNetwork)