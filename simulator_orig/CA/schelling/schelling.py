#coding:utf-8

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import random
import copy

class Schelling(object):
    #constructor
    def __init__(self,width,height,empty_ratio,
                 similarity_threshold,n_iterations,races = 2):
        self.width = width
        self.height = height
        self.empty_ratio = empty_ratio
        self.simirality_threshold = similarity_threshold
        self.n_iterations = n_iterations
        self.races = races
        self.empty_houses = []
        self.agents = {}
        self.list_for_graph = []
        self.ims = []

    #distributes people in the grid randomly
    def populate(self):
        self.all_houses = list(itertools.product(range(self.width),range(self.height)))
        random.shuffle(self.all_houses)
        self.n_empty = int(self.empty_ratio*len(self.all_houses))
        self.empty_houses = self.all_houses[:self.n_empty]
        self.remaining_houses = self.all_houses[self.n_empty:]
        houses_by_race = [self.remaining_houses[i::self.races] for i in range(self.races)]
        for i in range(self.races):
            self.agents = dict(
                self.agents.items() +
                dict(zip(houses_by_race[i],[i+1]*len(houses_by_race[i]))).items()
            )

    #check neighbors
    def is_unsatisfied(self,x,y):
        race = self.agents[(x,y)] #その座標での人種
        count_similar = 0
        count_different = 0
        #座標に応じて場合分け（８方向）
        if x > 0 and y > 0 and (x - 1, y - 1) not in self.empty_houses:
            if self.agents[(x-1,y-1)] == race:
                count_similar += 1
            else:
                count_different += 1
        if y > 0 and (x, y - 1) not in self.empty_houses:
            if self.agents[(x, y - 1)] == race:
                count_similar += 1
            else:
                count_different += 1
        if x < (self.width-1) and y > 0 and (x + 1, y - 1) not in self.empty_houses:
            if self.agents[(x + 1, y - 1)] == race:
                count_similar += 1
            else:
                count_different += 1
        if x > 0 and (x - 1, y) not in self.empty_houses:
            if self.agents[(x - 1, y)] == race:
                count_similar += 1
            else:
                count_different += 1
        if x < (self.width-1) and (x + 1, y) not in self.empty_houses:
            if self.agents[(x + 1, y)] == race:
                count_similar += 1
            else:
                count_different += 1
        if x > 0 and y < (self.height-1) and (x - 1, y + 1) not in self.empty_houses:
            if self.agents[(x - 1, y + 1)] == race:
                count_similar += 1
            else:
                count_different += 1
        if y < (self.height-1) and (x, y + 1) not in self.empty_houses:
            if self.agents[(x, y + 1)] == race:
                count_similar += 1
            else:
                count_different += 1
        if x <(self.width-1) and y < (self.height-1) and (x + 1, y + 1) not in self.empty_houses:
            if self.agents[(x + 1, y + 1)] == race:
                count_similar += 1
            else:
                count_different += 1

        if count_different+count_similar == 0:
            return False,0
        else:
            similarity = float(count_similar)/(count_similar+count_different)
            return similarity < self.simirality_threshold,similarity

    #move unsatisfied agents
    def update(self):
        for i in range(self.n_iterations):
            self.old_agents = copy.deepcopy(self.agents)
            n_changes = 0
            similarity_tmp = 0
            average_similarity = 0
            for agent in self.old_agents:
                is_unsatisfied = self.is_unsatisfied(agent[0],agent[1])
                similarity_tmp += is_unsatisfied[1]
                if(is_unsatisfied[0]):
                    agent_race = self.agents[agent]
                    # randomly choose the empty house, where agent will move
                    empty_house = random.choice(self.empty_houses)
                    self.agents[empty_house] = agent_race
                    del self.agents[agent]
                    self.empty_houses.remove(empty_house)
                    self.empty_houses.append(agent)
                    n_changes += 1
            self.list_for_graph.append(similarity_tmp/len(self.old_agents))
            print "iterations:%3d | # of changes:%4d | ave similarity:%9f" %(i,n_changes,similarity_tmp/len(self.old_agents))
            #stable state
            if n_changes == 0:
                break

    def move_to_empty(self,x,y):
        race = self.agents[(x,y)]
        empty_house = random.choice(self.empty_houses)
        self.updated_agents[empty_house] = race
        del self.updated_agents[(x,y)]
        self.empty_houses.remove(empty_house)
        self.empty_houses.append((x,y))

    # draw the whole city and people living in the city
    def plot(self,title,filename):
        fig, ax = plt.subplots()
        agent_colors = {1:'b',2:'r',3:'g',4:'c'}
        for agent in self.agents:
            ax.scatter(agent[0]+0.5,agent[1]+0.5,color=agent_colors[self.agents[agent]])
        #property of the figure
        ax.set_title(title,fontsize=10,fontweight='bold')
        ax.set_xlim([0,self.width])
        ax.set_ylim([0,self.height])
        plt.savefig(filename)

schelling_1 = Schelling(50,50,0.2,0.7,1000,2)
schelling_1.populate()
schelling_1.plot("schalling initial","schelling2_initial.png")
schelling_1.update()
schelling_1.plot("schalling updated","schelling2_updated.png")

#plot how the n_change decline with iterations
x = range(len(schelling_1.list_for_graph))
plt.plot(x,schelling_1.list_for_graph,'b-')
plt.show()
