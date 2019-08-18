import tensorflow as tf
import numpy as np
from animal import animalEnv, isvalid
import networkx as nx
import math
import time
import pickle
#import tkinter as tk

start = time.time()

#win = tk.Tk()
#win.title("闘獣棋")
#win.geometry("448x576")

#board_img = tk.PhotoImage(file="img/board.png")
"""
playerA_img = [tk.PhotoImage(file="img/mouse.png"), tk.PhotoImage(file="img/cat.png"),
               tk.PhotoImage(file="img/dog.png"), tk.PhotoImage(file="img/wolf.png"),
               tk.PhotoImage(file="img/panther.png"), tk.PhotoImage(file="img/lion.png"),
               tk.PhotoImage(file="img/tiger.png"), tk.PhotoImage(file="img/elephant.png")]
playerB_img = [tk.PhotoImage(file="img/mouse2.png"), tk.PhotoImage(file="img/cat2.png"),
               tk.PhotoImage(file="img/dog2.png"), tk.PhotoImage(file="img/wolf2.png"),
               tk.PhotoImage(file="img/panther2.png"), tk.PhotoImage(file="img/lion2.png"),
               tk.PhotoImage(file="img/tiger2.png"), tk.PhotoImage(file="img/elephant2.png")]
cv = tk.Canvas(win, width=448-1, height=576-1)
cv.place(x=0, y=0)
"""
gpuConfig = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
    device_count={'GPU': 0})

input_height = 9
input_width = 7
input_channels = 7
conv_n_maps = [64, 64]
conv_kernel_sizes = [(3, 3), (3, 3)]
conv_strides = [1, 1]
conv_paddings = ["SAME"] * 2
conv_activations = [tf.nn.relu] * 2

v_conv_n_map = 1
v_conv_kernel_size = 1
v_conv_stride = 1
v_conv_padding = "SAME"
v_conv_activation = tf.nn.relu
v_n_hidden = 64
v_hidden_activation = tf.nn.relu
v_n_output = 1

p_conv_n_maps = [64, 4]
p_conv_kernel_sizes = [(3, 3), (3, 3)]
p_conv_strides = [1, 1]
p_conv_paddings = ["SAME"] * 2
p_conv_activations = [tf.nn.relu] * 2

initializer = tf.variance_scaling_initializer()

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

def softmax(X):
    shape = tf.shape(X)
    X_exp = tf.exp(X)
    X_sum = tf.identity(X_exp)
    for i in [3, 2, 1]:
        X_sum = tf.reduce_sum(X_sum, axis=i)
    X_soft = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(X_sum, 1), 2) ,3), [1, shape[1], shape[2], shape[3]])
    return tf.divide(X_exp, X_soft)

with tf.variable_scope("alpha") as scope:
    conv0 = tf.layers.conv2d(X_state, filters=conv_n_maps[0], kernel_size=conv_kernel_sizes[0], strides=conv_strides[0],
                             padding=conv_paddings[0], activation=conv_activations[0], kernel_initializer=initializer)
    conv1 = tf.layers.conv2d(conv0, filters=conv_n_maps[1], kernel_size=conv_kernel_sizes[1], strides=conv_strides[1],
                             padding=conv_paddings[1], activation=conv_activations[1], kernel_initializer=initializer)                     
    v_conv = tf.layers.conv2d(conv1, filters=v_conv_n_map, kernel_size=v_conv_kernel_size, strides=v_conv_stride,
                              padding=v_conv_padding, activation=v_conv_activation, kernel_initializer=initializer)
    v_conv_flat = tf.reshape(v_conv, shape=[-1, input_height * input_width])
    v_hidden = tf.layers.dense(v_conv_flat, v_n_hidden, activation=v_hidden_activation, kernel_initializer=initializer)
    p_conv0 = tf.layers.conv2d(conv1, filters=p_conv_n_maps[0], kernel_size=p_conv_kernel_sizes[0], strides=p_conv_strides[0],
                               padding=p_conv_paddings[0], activation=p_conv_activations[0], kernel_initializer=initializer)
    p_conv1 = tf.layers.conv2d(p_conv0, filters=p_conv_n_maps[1], kernel_size=p_conv_kernel_sizes[1], strides=p_conv_strides[1],
                               padding=p_conv_paddings[1], activation=p_conv_activations[1], kernel_initializer=initializer)

    p = softmax(p_conv1)
    v = tf.layers.dense(v_hidden, v_n_output, activation=tf.nn.tanh, kernel_initializer=initializer)

saver = tf.train.Saver()

example_state = []
example_pi = []
example_z = []
example_memory = 5000 # 50000
env = animalEnv()

max_turn = 256
depth = 800 #800

batch_size = 32

class MCTS:
    
    def __init__(self):
        self.digraph = None
        self.node_count = None
        self.root = None
        self.turn = None

    def reset(self, c=1):
        self.digraph = nx.DiGraph()
        self.digraph.add_node(0, state=np.array([[-6,  0,  0,  0,  0,  0, -7],
                                                 [ 0, -3,  0,  0,  0, -2,  0],
                                                 [-1,  0, -5,  0, -4,  0, -8],
                                                 [ 0,  0,  0,  0,  0,  0,  0],
                                                 [ 0,  0,  0,  0,  0,  0,  0],
                                                 [ 0,  0,  0,  0,  0,  0,  0],
                                                 [ 8,  0,  4,  0,  5,  0,  1],
                                                 [ 0,  2,  0,  0,  0,  3,  0],
                                                 [ 7,  0,  0,  0,  0,  0,  6]], dtype=np.int32), 
                              W=0, N=1, P=1, A=None, done=0, player=1, turn=0)
        self.node_count = 1
        self.root = 0
        self.c_init = 1.25
        self.c_base = 19652
        self.turn = 0
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

    def add_exploration_noise(self, node_num):
        actions = self.digraph.out_degree(node_num)
        noise = np.random.gamma(self.root_dirichlet_alpha, 1, actions)
        frac = self.root_exploration_fraction
        for a, n in zip(self.digraph.successors(node_num), noise):
            self.digraph.nodes[a]["P"] = self.digraph.nodes[a]["P"] * (1 - frac) + n * frac

    def obs(self, node_num):
        obs0 = np.copy(self.digraph.nodes[node_num]["state"])
        if node_num == 0:
            return obs0, np.copy(obs0), np.copy(obs0)
        n_pre = node_num
        n3 = 0
        n7 = 0
        for i in range(7):
            n_pre = list(self.digraph.predecessors(n_pre))[0]
            if n_pre == 0:
                break
            if i == 2:
                n3 = n_pre
            if i == 6:
                n7 = n_pre
        return obs0, np.copy(self.digraph.nodes[n3]["state"]), np.copy(self.digraph.nodes[n7]["state"])

    def obs_pre(self, node_num):
        obs0 = np.copy(self.digraph.nodes[node_num]["state"])
        if node_num == 0:
            return np.copy(obs0), np.copy(obs0)
        n_pre = node_num
        n2 = 0
        n6 = 0
        for i in range(6):
            n_pre = list(self.digraph.predecessors(n_pre))[0]
            if n_pre == 0:
                break
            if i == 1:
                n2 = n_pre
            if i == 5:
                n6 = n_pre
        return np.copy(self.digraph.nodes[n2]["state"]), np.copy(self.digraph.nodes[n6]["state"])

    def step(self, node_num):
        if self.digraph.nodes[node_num]["done"] == 1:
            self.digraph.nodes[node_num]["W"] += self.digraph.nodes[node_num]["W"] / self.digraph.nodes[node_num]["N"]
            self.digraph.nodes[node_num]["N"] += 1
            return
        elif self.digraph.out_degree(node_num) == 0:
            pass_flag = True
            obs0, obs3, obs7 = self.obs(node_num)
            player = self.digraph.nodes[node_num]["player"]
            turn = self.digraph.nodes[node_num]["turn"]
            state = np.array([state_reshape(obs0, obs3, obs7, player)])
            prob = p.eval(feed_dict={X_state: state})[0]
            for y in range(input_height):
                for x in range(input_width):
                    for a in range(4):
                        obs, reward, done, valid = isvalid(obs0, obs3, obs7, y, x, a, player, turn)
                        if valid:
                            pass_flag = False
                            if done:
                                self.digraph.nodes[node_num]["W"] += reward
                                self.digraph.add_node(self.node_count, state=np.copy(obs), W=reward, N=1, P=prob[y][x][a], A=[y, x, a], done=1, player=-player, turn=turn+1)
                            else:
                                obs2, obs6 = self.obs_pre(node_num)
                                state_v = np.array([state_reshape(np.copy(obs), obs2, obs6, -player)])
                                val = v.eval(feed_dict={X_state: state_v})[0][0]
                                self.digraph.nodes[node_num]["W"] += val
                                self.digraph.add_node(self.node_count, state=np.copy(obs), W=val, N=1, P=prob[y][x][a], A=[y, x, a], done=0, player=-player, turn=turn+1)
                            self.digraph.add_edge(node_num, self.node_count)
                            self.node_count += 1

            if pass_flag:
                self.digraph.nodes[node_num]["N"] += 1
                self.digraph.add_node(self.node_count, state=self.digraph.nodes[node_num]["state"], W=0, N=1, P=1, A=None, done=0, player=-player, turn=turn+1)
                self.digraph.add_edge(node_num, self.node_count)
                self.node_count += 1

            N_sum = 0
            W_sum = 0
            for n in self.digraph.successors(node_num):
                N_sum += self.digraph.nodes[n]["N"]
                W_sum += self.digraph.nodes[n]["W"]
            self.digraph.nodes[node_num]["N"] = N_sum
            self.digraph.nodes[node_num]["W"] = W_sum

        else:
            self.add_exploration_noise(node_num)
            self.step(self.PUCT_rule(node_num))

            N_sum = 0
            W_sum = 0
            for n in self.digraph.successors(node_num):
                N_sum += self.digraph.nodes[n]["N"]
                W_sum += self.digraph.nodes[n]["W"]
            self.digraph.nodes[node_num]["N"] = N_sum
            self.digraph.nodes[node_num]["W"] = W_sum

        return
    
    def PUCT_rule(self, node_num):
        Np = self.digraph.nodes[node_num]["N"]
        max_PUCT = -np.inf
        next_node = None
        player = self.digraph.nodes[node_num]["player"]
        for n in self.digraph.successors(node_num):
            node = self.digraph.nodes[n]
            C = math.log((Np + self.c_base + 1) / self.c_base) + self.c_init
            u_score = C * node["P"] * math.sqrt(Np) / (1 + node["N"])
            q_score = node["W"]*player/node["N"]
            PUCT = q_score + u_score
            if max_PUCT < PUCT:
                max_PUCT = PUCT
                next_node = n

        return next_node

    def search(self):
        self.step(self.root)

    def move(self):
        pi = np.zeros((input_height, input_width, 4))
        max_N = 0
        next_node = None
        for cand in self.digraph.successors(self.root):
            cand_node = self.digraph.nodes[cand]
            if cand_node["A"] == None:
                next_node = cand
                break
            y = cand_node["A"][0]
            x = cand_node["A"][1]
            a = cand_node["A"][2]
            N = cand_node["N"]
            pi[y][x][a] = N
            if N > max_N:
                max_N = N
                next_node = cand
        if pi.sum() == 0:
            pi = None
        """
        for n in self.digraph.nodes:
            print(self.digraph.nodes[n])
        """
        self.root = next_node
        self.turn += 1
        pi /= pi.sum()
        action = self.digraph.nodes[next_node]["A"]
        return action[0], action[1], action[2], pi

def state_reshape(obs0, obs3, obs7, player):
    obs0_x = np.copy(obs0.reshape(9, 7, 1))
    obs0_x[obs0_x<=0] = 0
    obs0_o = np.copy(obs0.reshape(9, 7, 1))
    obs0_o *= -1
    obs0_o[obs0_o<=0] = 0
    obs3_x = np.copy(obs3.reshape(9, 7, 1))
    obs3_x[obs3_x<=0] = 0
    obs3_o = np.copy(obs3.reshape(9, 7, 1))
    obs3_o *= -1
    obs3_o[obs3_o<=0] = 0
    obs7_x = np.copy(obs7.reshape(9, 7, 1))
    obs7_x[obs7_x<=0] = 0
    obs7_o = np.copy(obs7.reshape(9, 7, 1))
    obs7_o *= -1
    obs7_o[obs7_o<=0] = 0

    if player==1:
        return np.concatenate((obs0_x, obs0_o, obs3_x, obs3_o, obs7_x, obs7_o, np.ones((9, 7, 1))), axis=2)
    else:
        return np.concatenate((obs0_x, obs0_o, obs3_x, obs3_o, obs7_x, obs7_o, -np.ones((9, 7, 1))), axis=2)

"""
def draw(obs):
    cv.delete("board")
    cv.create_image(0, 0, image=board_img, anchor="nw", tag="board")
    for y in range(9):
        for x in range(7):
            tag = str(y) + str(x)
            cv.delete(tag)
            if obs[y][x] > 0:
                cv.create_image(64*x, 64*y, image=playerA_img[abs(obs[y][x])-1], anchor="nw", tag=tag)
            if obs[y][x] < 0:
                cv.create_image(64*x, 64*y, image=playerB_img[abs(obs[y][x])-1], anchor="nw", tag=tag)
    win.update()
    #import time
    #time.sleep(10)
    return
"""

def player():
    valid = False
    while not valid:
        print("input your action")
        board, _, _, _ = env.get_state()
        print(board)
        y, x, a = map(int, input().split(" "))
        reward, done, valid = env.step(y, x, a)
    return reward, done

def com():
    for i in range(depth):
        Tree.search()
    y_out, x_out, a_out, pi_out = Tree.move()
    if type(pi_out) != np.ndarray:
        env.player *= -1
        return 0, False
    reward, done, _ = env.step(y_out, x_out, a_out)

    return reward, done

if __name__=="__main__":
    with tf.Session(config=gpuConfig) as sess:
        saver.restore(sess, "./my_dqn_gui.ckpt")
        Tree = MCTS()
        Tree.reset()
        env.reset()
        for i in range(max_turn):
            if env.player==1:
                reward, done = player()
            else:
                reward, done = com()
            if done:
                break
        if reward==1:
            print("player won")
        elif reward==-1:
            print("com won")
        else:
            print("draw")