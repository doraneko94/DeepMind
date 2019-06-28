import tensorflow as tf
import numpy as np
from ox import oxEnv, isvalid
import networkx as nx
import math
import time
import pickle

start = time.time()

gpuConfig = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
    device_count={'GPU': 0})

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
        p = p_raw / tf.reduce_sum(p_raw)
        v = tf.layers.dense(hidden2, n_outputs2, activation=hidden_activation2, kernel_initializer=initializer, kernel_regularizer=regularizer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return p, v, trainable_vars_by_name

p_new, v_new, vars_new = alpha(X_state, "alpha/new")

learning_rate = 0.00001

with tf.variable_scope("train"):
    z = tf.placeholder(tf.float32, shape=[None, 1])
    pi = tf.placeholder(tf.float32, shape=[None, 9])

    error = tf.square(z - v_new)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pi , logits=p_new)
    loss = tf.reduce_mean(error + entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

example_state = []
example_pi = []
example_z = []
example_memory = 5000 # 50000
env = oxEnv()

max_turn = 9
depth = 9

batch_size = 32

class MCTS:
    
    def __init__(self, c=1):
        self.digraph = nx.DiGraph()
        self.digraph.add_node(0, state=np.zeros(9, dtype=np.int32), W=0, N=0, P=1, A=None, done=0, player=1, lock=0, turn=0)
        self.node_count = 1
        self.root = 0
        self.c = c
        self.dc = 0.9
        self.dgamma = 0.9
        self.turn = 0
        self.kifu = {}

    def reset(self, c=1):
        self.root = 0
        self.c = c
        self.turn = 0
        self.kifu = {}

    def update_p(self):
        for node_num in self.digraph.nodes:
            if self.digraph.nodes[node_num]["lock"] != 1:
                obs0, obs1, obs3 = self.obs(node_num)
                state = np.array([state_reshape(obs0, obs1, obs3, self.digraph.nodes[node_num]["player"])])
                prob = p_new.eval(feed_dict={X_state: state})[0]
                for n in self.digraph.successor(node_num):
                    self.digraph.nodes[n]["P"] = prob[self.digraph.nodes[n]["A"]]

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
            self.digraph.nodes[node_num]["W"] += self.digraph.nodes[node_num]["W"] / self.digraph.nodes[node_num]["N"]
            self.digraph.nodes[node_num]["N"] += 1
            return
        elif self.digraph.out_degree(node_num) == 0:
            pass_flag = True
            obs0, obs1, obs3 = self.obs(node_num)
            state = np.array([state_reshape(obs0, obs1, obs3, self.digraph.nodes[node_num]["player"])])
            prob = p_new.eval(feed_dict={X_state: state})[0]
            lock_lst = []
            node_lst = []
            for a in range(9):
                obs, reward, done, valid = isvalid(self.digraph.nodes[node_num]["state"], a, self.digraph.nodes[node_num]["player"])
                if valid:
                    node_lst.append(self.node_count)
                    pass_flag = False
                    if done:
                        self.digraph.nodes[node_num]["W"] += reward
                        self.digraph.add_node(self.node_count, state=obs, W=reward, N=1, P=prob[a], A=a, done=1, lock=1, player=-self.digraph.nodes[node_num]["player"])
                        if reward == self.digraph.nodes[node_num]["player"]:
                            self.digraph.nodes[node_num]["lock"] = 1
                            lock_lst.append(a)
                    else:
                        obs0 = np.copy(obs)
                        obs1, obs3 = self.obs_pre(node_num)
                        state_v = np.array([state_reshape(obs0, obs1, obs3, self.digraph.nodes[node_num]["player"])])
                        v = v_new.eval(feed_dict={X_state: state_v})[0][0]
                        self.digraph.nodes[node_num]["W"] += v
                        self.digraph.add_node(self.node_count, state=obs, W=v, N=1, P=prob[a], A=a, done=0, lock=0, player=-self.digraph.nodes[node_num]["player"])
                    self.digraph.add_edge(node_num, self.node_count)
                    self.node_count += 1

            if len(lock_lst) > 0:
                prob = np.zeros(9)
                for i in lock_lst:
                    prob[i] = 1
                prob = prob / prob.sum()
                for i in node_lst:
                    self.digraph.nodes[i]["P"] = prob[self.digraph.nodes[i]["A"]]

            if pass_flag:
                self.digraph.nodes[node_num]["N"] += 1
                self.digraph.add_node(self.node_count, state=self.digraph.nodes[node_num]["state"], W=0, N=1, P=1, A=None, done=0, lock=1, player=-self.digraph.nodes[node_num]["player"])
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
            if self.root > node_num:
                self.step(self.kifu[node_num])
            else:
                self.step(self.PUCT_rule(node_num))
            W_sum = 0
            N_sum = 0
            for n in self.digraph.successors(node_num):
                N_sum += self.digraph.nodes[n]["N"]
                W_sum += self.digraph.nodes[n]["W"]
            self.digraph.nodes[node_num]["N"] = N_sum
            self.digraph.nodes[node_num]["W"] = W_sum

        return
    
    def PUCT_rule(self, node_num):
        rNp = math.sqrt(self.digraph.nodes[node_num]["N"])
        PUCT = np.zeros(9)
        ns = np.zeros(9)

        obs0, obs1, obs3 = self.obs(node_num)
        state = np.array([state_reshape(obs0, obs1, obs3, self.digraph.nodes[node_num]["player"])])
        prob = p_new.eval(feed_dict={X_state: state})[0]
        for n in self.digraph.successors(node_num):
            node = self.digraph.nodes[n]
            ns[node["A"]] = n 
            node["P"] = prob[node["A"]]
            PUCT[node["A"]] = node["W"]*self.digraph.nodes[node_num]["player"]/node["N"] + (self.c*(self.dc**self.turn))*node["P"]*rNp/(1+node["N"])
        if PUCT.sum() == 0:
            PUCT += 1
        PUCT = PUCT / PUCT.sum()

        next_node = ns[PUCT==PUCT.max()][0]
        return next_node

    def search(self):
        self.step(0)

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

        next_node = ns[pi==pi.max()][0]
        self.kifu[self.root] = next_node
        self.root = next_node
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

def example(example_memory, gamma=0.9, c=0.5):
    ex_s = []
    ex_p = []
    Tree.reset(c)
    env.reset()
    z_out = 0
    for _ in range(max_turn):
        for _ in range(depth):
            Tree.search()
        action_out, pi_out = Tree.move()
        obs0, obs1, obs3, reward, done, player, _ = env.step(action_out)

        s_out = state_reshape(obs0, obs1, obs3, player)
        
        ex_s.append(s_out)
        ex_p.append(pi_out)

        if done:
            z_out = reward
            break
    ex_z = [z_out*gamma**i for i in range(len(ex_p))][::-1]
    for ss, pp, zz in zip(ex_s, ex_p, ex_z):
        if len(example_z) == example_memory:
            index = np.random.randint(example_memory)
            example_state[index] = ss
            example_pi[index] = pp
            example_z[index] = zz
        else:
            example_state.append(ss)
            example_pi.append(pp)
            example_z.append(zz)

def make_batch(batch_size):
    indices = np.random.randint(0, len(example_z), size=[batch_size])
    state_batch = np.array([example_state[i] for i in indices])
    pi_batch = np.array([example_pi[i] for i in indices])
    z_batch = np.array([[example_z[i]] for i in indices])
    return state_batch, pi_batch, z_batch

def battle_with_rand(show=False):
    env.reset()
    player = 1
    done = False
    obs0 = np.zeros((1,9))
    obs1 = np.zeros((1,9))
    obs3 = np.zeros((1,9))
    if np.random.random() < 0.5:
        new_player = 1
    else:
        new_player = -1
    if show:
        print(new_player)
    while (not done):
        if show:
            print(obs0.reshape((3,3)))
            print("player: {}".format(player))
        state = np.array([state_reshape(obs0, obs1, obs3, player)])
        if new_player == player:
            prob = p_new.eval(feed_dict={X_state: state})
            if show:
                print(prob)
        else:
            prob = np.random.random((1, 9))
        actions = np.argsort(prob[0])[::-1]
        for a in actions:
            obs0, obs1, obs3, reward, done, player, valid = env.step(a)
            if valid:
                break
    if show:
        print(reward)
    return reward, new_player

c = 1.25

if __name__=="__main__":
    with tf.Session(config=gpuConfig) as sess:
        init.run()
        Tree = MCTS(c)
        for i in range(301): # 3001
            for j in range(25): # 100
                example(example_memory)
            for j in range(25): # 100
                state_batch, pi_batch, z_batch = make_batch(batch_size)
                if j == 0:
                    print("{}: loss={}".format(i, loss.eval(feed_dict={X_state: state_batch, z: z_batch, pi: pi_batch})))
            if i % 100 == 0:
                ent1 = entropy.eval(feed_dict={X_state: state_batch, pi: pi_batch})
                print(ent1)
            training_op.run(feed_dict={X_state: state_batch, z: z_batch, pi: pi_batch})
            
            if i % 100 == 0:
                saver.save(sess, "./my_dqn.ckpt")
                with open("Tree.pkl", "wb") as f:
                    pickle.dump(Tree, f)
            
            if i >= 1000:
                learning_rate = 0.02
            if i >= 2000:
                learning_rate = 0.002

        win_x = 0
        win_o = 0
        lose_x = 0
        lose_o = 0
        for i in range(100):
            if i % 10 == 0:
                reward, new_player = battle_with_rand(True)
            else:
                reward, new_player = battle_with_rand()
            if reward == -1:
                if new_player == -1:
                    win_o += 1
                else:
                    lose_o += 1
            if reward == 1:
                if new_player == 1:
                    win_x += 1
                else:
                    lose_x += 1
    if win_x + win_o + lose_x + lose_o == 0:
        print("None")
    else:
        print("final win rate: {}".format((win_x + win_o)/(win_x + win_o + lose_x + lose_o)))
        if win_x + lose_x == 0:
            print("\twin rate for  1: None")
        else:
            print("\twin rate for  1: {}".format((win_x)/(win_x + lose_x)))
        if win_o + lose_o == 0:
            print("\twin rate for -1: None")
        else:
            print("\twin rate for -1: {}".format((win_o)/(win_o + lose_o)))

end = 0
for i in Tree.digraph.nodes:
    if Tree.digraph.node[i]["done"] == 1:
        end += 1
print("end: {}".format(end))
print("time: {}".format(time.time() - start))