import numpy as np
from collections import deque
import copy

class oxEnv:

    def __init__(self):
        self.board = None
        self.obsque = None
        self.current_player = None
        self.result = None

    def reset(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.obsque = deque([np.zeros(9, dtype=np.int32)] * 3)
        self.current_player = 1

    def step(self, index):
        if self.board[index] != 0:
            return None, None, None, None, None, None, 0
        elif self.current_player == 1:
            self.board[index] = 1
            self.current_player = -1
        else:
            self.board[index] = -1
            self.current_player = 1
        obs0 = np.copy(self.board)
        obs3 = np.copy(self.obsque[0])
        obs1 = np.copy(self.obsque[2])
        self.obsque.popleft()
        self.obsque.append(np.copy(obs0))
        reward, done = self.check_game_result()
        valid = 1
        return obs0, obs1, obs3, reward, done, -self.current_player, valid

    def check_game_result(self):
        x_win, o_win, is_full = False, False, False

        for i in range(3):
            row = self.board[(i * 3):(i * 3 + 3)]
            col = self.board[i::3]
            if np.sum(row) == 3 or np.sum(col) == 3:
                x_win = True
            if np.sum(row) == -3 or np.sum(col) == -3:
                o_win = True
        if np.sum(self.board[[0, 4, 8]]) == 3 or np.sum(self.board[[2, 4, 6]]) == 3:
            x_win = True
        if np.sum(self.board[[0, 4, 8]]) == -3 or np.sum(self.board[[2, 4, 6]]) == -3:
            o_win = True
        if 0 not in self.board:
            is_full = True
        if x_win:
            reward = 1
        elif o_win:
            reward = -1
        else:
            reward = 0
        done = x_win or o_win or is_full
        return reward, done

def isvalid(state, action, player):
    board = np.copy(state)
    if board[action] != 0:
        return None, None, None, 0
    elif player == 1:
        board[action] = 1
    else:
        board[action] = -1
    obs0 = board
    reward, done = check_game_result(board)
    return obs0, reward, done, 1

def check_game_result(board):
    x_win, o_win, is_full = False, False, False

    for i in range(3):
        row = board[(i * 3):(i * 3 + 3)]
        col = board[i::3]
        if np.sum(row) == 3 or np.sum(col) == 3:
            x_win = True
        if np.sum(row) == -3 or np.sum(col) == -3:
            o_win = True
    if np.sum(board[[0, 4, 8]]) == 3 or np.sum(board[[2, 4, 6]]) == 3:
        x_win = True
    if np.sum(board[[0, 4, 8]]) == -3 or np.sum(board[[2, 4, 6]]) == -3:
        o_win = True
    if 0 not in board:
        is_full = True
    if x_win:
        reward = 1
    elif o_win:
        reward = -1
    else:
        reward = 0
    done = x_win or o_win or is_full
    return reward, done

if __name__=="__main__":
    env = oxEnv()
    env.reset()
    for _ in range(20):
        a = int(input())
        obs0, obs1, obs3, reward, done, player, valid = env.step(a)
        if not valid:
            continue
        print(obs1.reshape((3, -1)))
        if done:
            break