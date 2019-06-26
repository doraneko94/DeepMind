from MCTS import MCTS
from ox import oxEnv
import numpy as np

env = oxEnv()
env.reset()

Tree = MCTS()
example_s = []
example_p = []
z = 0
for i in range(9):
    for j in range(3):
        Tree.search()
    action, pi = Tree.move()
    obs0, obs1, obs3, reward, done, player, valid = env.step(action)
    print(obs0, obs1, obs3, reward, done, player, valid)

    obs0_x = np.copy(obs0.reshape(3,3))
    obs0_x[obs0_x!=1] = 0
    obs0_o = np.copy(obs0.reshape(3,3))
    obs0_o[obs0_o!=-1] = 0
    obs1_x = np.copy(obs1.reshape(3,3))
    obs1_x[obs1_x!=1] = 0
    obs1_o = np.copy(obs1.reshape(3,3))
    obs1_o[obs1_o!=-1] = 0
    obs3_x = np.copy(obs3.reshape(3,3))
    obs3_x[obs0_x!=1] = 0
    obs3_o = np.copy(obs3.reshape(3,3))
    obs3_o[obs0_o!=-1] = 0

    if player==1:
        s = np.array([obs0_x, obs0_o, obs1_x, obs1_o, obs3_x, obs3_o, np.ones((3,3))])
    else:
        s = np.array([obs0_x, obs0_o, obs1_x, obs1_o, obs3_x, obs3_o, -np.ones((3,3))])

    example_s.append(s)
    example_p.append(pi)

    if done:
        z = reward
        break
print(example_p[-1])