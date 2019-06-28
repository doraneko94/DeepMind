import tensorflow as tf
import numpy as np
from ox import oxEnv, isvalid
import networkx as nx
import math

input_height = 3
input_width = 3
input_channels = 7
conv_n_maps = [4, 8]
conv_kernel_sizes = [(2, 2), (1, 1)]
conv_strides = [1, 1]
conv_paddings = ["VALID"] * 2
conv_activation = [tf.nn.relu] * 2

n_hidden_in = 2 * 2 * 8
n_hidden = 16
hidden_activation1 = tf.nn.sigmoid
hidden_activation2 = tf.nn.tanh
n_outputs1 = 9
n_outputs2 = 1
scale = 0.001
initializer = tf.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale)

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

def alpha(X_state, name):
    prev_layer = X_state

    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation
            ):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer, kernel_regularizer=regularizer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden1 = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        hidden2 = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        hidden3 = tf.layers.dense(hidden1, n_hidden,
                                  activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        p_raw = tf.layers.dense(hidden3, n_outputs1, activation=hidden_activation1, kernel_initializer=initializer, kernel_regularizer=regularizer)
        p = tf.nn.softmax(p_raw)
        v = tf.layers.dense(hidden2, n_outputs2, activation=hidden_activation2, kernel_initializer=initializer, kernel_regularizer=regularizer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return p, v, trainable_vars_by_name

p_old, v_old, vars_old = alpha(X_state, "alpha/old")
p_new, v_new, vars_new = alpha(X_state, "alpha/new")

copy_ops_no = [v_old.assign(vars_new[var_name])
            for var_name, v_old in vars_old.items()]
copy_new_to_old = tf.group(*copy_ops_no)
copy_ops_on = [v_new.assign(vars_old[var_name])
            for var_name, v_new in vars_new.items()]
copy_old_to_new = tf.group(*copy_ops_on)

learning_rate = 0.1
#momentum = 0.95

with tf.variable_scope("train"):
    z = tf.placeholder(tf.float32, shape=[None, 1])
    pi = tf.placeholder(tf.float32, shape=[None, 9])

    error = tf.square(z - v_new)
    p_soft = tf.nn.softmax(p_new)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi , logits=p_soft)
    loss = tf.reduce_mean(error + entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

example_state = []
example_pi = []
example_z = []
example_memory = 50000
env = oxEnv()

max_turn = 9
depth = 9

batch_size = 32

class MCTS:
    
    def __init__(self, c=1):
        self.digraph = nx.DiGraph()
        self.digraph.add_node(0, state=np.zeros(9, dtype=np.int32), W=0, N=0, P=1, A=None, done=0)
        self.node_count = 1
        self.root = 0
        self.init_player = 1
        self.c = c
        self.dc = 0.9
        self.dgamma = 0.9
        self.turn = 0

    def obs(self, node_num):
        obs0 = np.copy(self.digraph.nodes[node_num]["state"])
        pre1_lst = list(self.digraph.predecessors(node_num))
        if len(pre1_lst) == 0:
            return obs0, np.zeros((3,3)), np.zeros((3,3))
        obs1 = np.copy(self.digraph.nodes[pre1_lst[0]]["state"])
        pre2_lst = list(self.digraph.predecessors(pre1_lst[0]))
        if len(pre2_lst) == 0:
            return obs0, obs1, np.zeros((3,3))
        pre3_lst = list(self.digraph.predecessors(pre2_lst[0]))
        if len(pre3_lst) == 0:
            return obs0, obs1, np.zeros((3,3))
        obs3 = np.copy(self.digraph.nodes[pre3_lst[0]]["state"])
        return obs0, obs1, obs3

    def obs_pre(self, node_num):
        obs1 = np.copy(self.digraph.nodes[node_num]["state"])
        pre2_lst = list(self.digraph.predecessors(node_num))
        if len(pre2_lst) == 0:
            return obs1, np.zeros((3,3))
        pre3_lst = list(self.digraph.predecessors(pre2_lst[0]))
        if len(pre3_lst) == 0:
            return obs1, np.zeros((3,3))
        obs3 = np.copy(self.digraph.nodes[pre3_lst[0]]["state"])
        return obs1, obs3

    def step(self, node_num):
        if self.digraph.nodes[node_num]["done"] == 1:
            return #self.digraph.nodes[node_num]["W"]
        elif self.digraph.out_degree(node_num) == 0:
            pass_flag = True
            obs0, obs1, obs3 = self.obs(node_num)
            state = np.array([state_reshape(obs0, obs1, obs3, self.player)])
            prob = p_old.eval(feed_dict={X_state: state})[0]
            for a in range(9):
                obs, reward, done, valid = isvalid(self.digraph.nodes[node_num]["state"], a, self.player)
                if valid:
                    pass_flag = False
                    if done:
                        self.digraph.nodes[node_num]["W"] += reward
                        self.digraph.add_node(self.node_count, state=obs, W=reward, N=1, P=prob[a], A=a, done=1)
                    else:
                        obs0 = np.copy(obs)
                        obs1, obs3 = self.obs_pre(node_num)
                        state_v = np.array([state_reshape(obs0, obs1, obs3, self.player)])
                        v = v_old.eval(feed_dict={X_state: state_v})[0]
                        self.digraph.nodes[node_num]["W"] += v
                        self.digraph.add_node(self.node_count, state=obs, W=v, N=1, P=prob[a], A=a, done=0)
                    self.digraph.add_edge(node_num, self.node_count)
                    self.node_count += 1

            if pass_flag:
                self.digraph.nodes[node_num]["N"] += 1
                self.digraph.add_node(self.node_count, state=self.digraph.nodes[node_num]["state"], W=0, N=1, P=1, A=None, done=0)
                self.digraph.add_edge(node_num, self.node_count)
                self.node_count += 1

            W_sum = 0
            N_sum = 0
            for n in self.digraph.successors(node_num):
                N_sum += self.digraph.nodes[n]["N"]
                W_sum += self.digraph.nodes[n]["W"]
            self.digraph.nodes[node_num]["N"] = N_sum
            self.digraph.nodes[node_num]["W"] = W_sum

        else:
            self.step(self.PUCT_rule(node_num))
            W_sum = 0
            N_sum = 0
            for n in self.digraph.successors(node_num):
                N_sum += self.digraph.nodes[n]["N"]
                W_sum += self.digraph.nodes[n]["W"]
            self.digraph.nodes[node_num]["N"] = N_sum
            self.digraph.nodes[node_num]["W"] = W_sum

        return #self.digraph.nodes[node_num]["W"]
    
    def PUCT_rule(self, node_num):
        rNp = math.sqrt(self.digraph.nodes[node_num]["N"])
        PUCT = np.zeros(9)
        ns = np.zeros(9)
        #Q = np.zeros(9)
        #C = np.zeros(9)
        N = np.zeros(9)
        for n in self.digraph.successors(node_num):
            node = self.digraph.nodes[n]
            N[node["A"]] = node["N"]
            ns[node["A"]] = n 
            #Q[node["A"]] = node["W"]*self.player/node["N"]
            #C[node["A"]] = (self.c*(self.dc**self.turn))*node["P"]*rNp/(1+node["N"])
            #PUCT[node["A"]] = Q[node["A"]] + C[node["A"]]
            PUCT[node["A"]] = node["W"]*self.player/node["N"] + (self.c*(self.dc**self.turn))*node["P"]*rNp/(1+node["N"])
        if PUCT.sum() == 0:
            PUCT += 1
        PUCT = PUCT / PUCT.sum()
        #print(PUCT)
        #print("Q")
        #print(Q)
        #print("C")
        #print(C)
        print("N")
        print(N)
        next_node = ns[PUCT==PUCT.max()][0]
        self.player *= -1
        return next_node

    def search(self):
        self.player = self.init_player
        self.step(self.root)

    def move(self):
        pi = np.zeros(9)
        ns = np.zeros(9)
        for cand in self.digraph.successors(self.root):
            cand_node = self.digraph.nodes[cand]
            ns[cand_node["A"]] = cand
            pi[cand_node["A"]] = cand_node["N"]**(1/(self.dgamma**self.turn))
        if pi.sum() == 0:
            pi = np.ones(9)
        pi = pi / pi.sum()
        #print(ns)
        #print(pi)
        next_node = ns[pi==pi.max()][0]
        self.root = next_node
        self.init_player *= -1
        self.turn += 1
        return self.digraph.nodes[next_node]["A"], pi

def state_reshape(obs0, obs1, obs3, player):
    obs0_x = np.copy(obs0.reshape(3,3,1))
    obs0_x[obs0_x!=1] = 0
    obs0_o = np.copy(obs0.reshape(3,3,1))
    obs0_o[obs0_o!=-1] = 0
    obs1_x = np.copy(obs1.reshape(3,3,1))
    obs1_x[obs1_x!=1] = 0
    obs1_o = np.copy(obs1.reshape(3,3,1))
    obs1_o[obs1_o!=-1] = 0
    obs3_x = np.copy(obs3.reshape(3,3,1))
    obs3_x[obs0_x!=1] = 0
    obs3_o = np.copy(obs3.reshape(3,3,1))
    obs3_o[obs0_o!=-1] = 0

    if player==1:
        return np.concatenate((obs0_x, obs0_o, obs1_x, obs1_o, obs3_x, obs3_o, np.ones((3,3,1))), axis=2)
    else:
        return np.concatenate((obs0_x, obs0_o, obs1_x, obs1_o, obs3_x, obs3_o, -np.ones((3,3,1))), axis=2)

c = 1.25

if __name__=="__main__":
    with tf.Session() as sess:
        init.run()
        Tree = MCTS(c)
        for j in range(50):
            Tree.search()
    import pickle
    with open("Tree.pkl", "wb") as f:
        pickle.dump(Tree.digraph, f)
    