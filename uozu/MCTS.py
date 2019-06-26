import networkx as nx
import numpy as np
from ox import isvalid
import math

class MCTS:
    
    def __init__(self):
        self.digraph = nx.DiGraph()
        self.digraph.add_node(0, state=np.zeros(9, dtype=np.int32), W=0, N=0, P=1, A=None, done=0)
        self.node_count = 1
        self.root = 0
        self.init_player = 1
        self.c = 0.01

    def step(self, node_num):
        if self.digraph.nodes[node_num]["done"] == 1:
            return self.digraph.nodes[node_num]["W"]
        elif self.digraph.out_degree(node_num) == 0:
            pass_flag = True
            for a in range(9):
                obs, reward, done, valid = isvalid(self.digraph.nodes[node_num]["state"], a, self.player)
                if valid:
                    pass_flag = False
                    if done:
                        self.digraph.nodes[node_num]["W"] += reward
                        self.digraph.add_node(self.node_count, state=obs, W=reward, N=1, P=np.random.rand(), A=a, done=1)
                    else:
                        v = np.random.rand()
                        self.digraph.nodes[node_num]["W"] += v
                        self.digraph.add_node(self.node_count, state=obs, W=v, N=1, P=np.random.rand(), A=a, done=0)
                    self.digraph.add_edge(node_num, self.node_count)
                    self.node_count += 1

            if pass_flag:
                self.digraph.add_node(self.node_count, state=self.digraph.nodes[node_num]["state"], W=0, N=1, P=np.random.rand(), A=None, done=0)
                self.digraph.add_edge(node_num, self.node_count)
                self.node_count += 1
        else:
            self.digraph.nodes[node_num]["W"] += self.step(self.PUCT_rule(node_num))

        return self.digraph.nodes[node_num]["W"]
    
    def PUCT_rule(self, node_num, return_pi=False):
        rNp = math.sqrt(self.digraph.nodes[node_num]["N"])
        PUCT = np.zeros(9)
        ns = np.zeros(9)
        for n in self.digraph.successors(node_num):
            node = self.digraph.nodes[n]
            ns[node["A"]] = n 
            PUCT[node["A"]] = node["W"]*self.player/node["N"] + self.c*node["P"]*rNp/(1+node["N"])
        PUCT = PUCT / PUCT.sum()
        next_node = ns[PUCT==PUCT.max()][0]
        self.player *= -1
        if return_pi:
            return next_node, PUCT
        else:
            return next_node

    def search(self):
        self.player = self.init_player
        self.step(self.root)

    def move(self):
        next_node, pi = self.PUCT_rule(self.root, return_pi=True)
        self.root = next_node
        self.init_player *= -1
        return self.digraph.nodes[next_node]["A"], pi