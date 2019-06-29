import numpy as np
from collections import deque
import copy

class animalEnv:

    def __init__(self):
        self.board = None
        self.obsque = None
        self.player = None
        self.turn = None
        self.n_koma = None

    def reset(self):
        self.board = np.array([[-6,  0,  0,  0,  0,  0, -7],
                               [ 0, -3,  0,  0,  0, -2,  0],
                               [-1,  0, -5,  0, -4,  0, -8],
                               [ 0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0],
                               [ 0,  0,  0,  0,  0,  0,  0],
                               [ 8,  0,  4,  0,  5,  0,  1],
                               [ 0,  2,  0,  0,  0,  3,  0],
                               [ 7,  0,  0,  0,  0,  0,  6]], dtype=np.int32)
        self.obsque = deque([np.copy(self.board)] * 8)
        self.player = 1
        self.turn = 0
        self.n_koma = [8, 8] # -1, 1

    def get_state(self):
        return np.copy(self.board), np.copy(self.obsque[4]), np.copy(self.obsque[0]), self.player

    def step(self, y, x, direction):
        valid = 0
        nx = x
        ny = y
        if direction == 0:
            ny -= 1
        if direction == 1:
            nx += 1
        if direction == 2:
            ny += 1
        if direction == 3:
            nx -= 1
        if (nx < 0 or ny < 0 or nx >= 7 or ny >= 9):
            return None, None, valid
        if self.board[y][x] * self.player <= 0:
            return None, None, valid
        if (self.player, ny, nx) in [(1, 8, 3), (-1, 0, 3)]:
            return None, None, valid
        if abs(self.board[y][x]) == 1:
            if abs(self.board[ny][nx]) in [2, 3, 4, 5, 6, 7]:
                if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                    (8, 2), (7, 3), (8, 4)]:
                    return None, None, valid
        if abs(self.board[y][x]) in [2, 3, 4, 5, 8]:
            if (ny, nx) in [(3, 1), (3, 2), (3, 4), (3, 5),
                            (4, 1), (4, 2), (4, 4), (4, 5),
                            (5, 1), (5, 2), (5, 4), (5, 5)]:
                return None, None, valid
            if abs(self.board[ny][nx]) > abs(self.board[y][x]):
                if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                    (8, 2), (7, 3), (8, 4)]:
                    return None, None, valid
        if abs(self.board[y][x]) in [6, 7]:
            if (ny, nx) in [(3, 1), (3, 2), (3, 4), (3, 5),
                            (4, 1), (4, 2), (4, 4), (4, 5),
                            (5, 1), (5, 2), (5, 4), (5, 5)]:
                if direction == 0:
                    ny -= 3
                if direction == 1:
                    nx += 2
                if direction == 2:
                    ny += 3
                if direction == 3:
                    nx -= 2
            if abs(self.board[ny][nx]) > abs(self.board[y][x]):
                if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                    (8, 2), (7, 3), (8, 4)]:
                    return None, None, valid
        if self.board[y][x] * self.board[ny][nx] > 0:
            return None, None, valid
        valid = 1
        if self.board[ny][nx] < 0:
            self.n_koma[0] -= 1
        if self.board[ny][nx] > 0:
            self.n_koma[1] -= 1
        
        self.board[ny][nx] = self.board[y][x]
        self.board[y][x] = 0
        reward, done = self.check_game_result()

        self.obsque.popleft()
        self.obsque.append(np.copy(self.board))
        
        self.turn += 1
        self.player *= -1

        return reward, done, valid

    def check_game_result(self):
        if self.board[0][3] > 0:
            return 1, True
        if self.board[8][3] < 0:
            return -1, True
        if self.n_koma[0] <= 0:
            return 1, True
        if self.n_koma[1] <= 0:
            return -1, True
        if self.turn >= 200:
            return 0, True
        if (self.board == self.obsque[0]).all() and (self.board == self.obsque[4]).all():
            return 0, True
        return 0, False

def isvalid(board, obs4, obs8, y, x, direction, player, turn):
    state = np.copy(board)
    valid = False
    if state[y][x] * player <= 0:
        return None, None, None, valid
    nx = x
    ny = y
    if direction == 0:
        ny -= 1
    if direction == 1:
        nx += 1
    if direction == 2:
        ny += 1
    if direction == 3:
        nx -= 1
    if (nx < 0 or ny < 0 or nx >= 7 or ny >= 9):
        return None, None, None, valid
    if (player, ny, nx) in [(1, 8, 3), (-1, 0, 3)]:
        return None, None, None, valid
    if abs(state[y][x]) == 1:
        if abs(state[ny][nx]) in [2, 3, 4, 5, 6, 7]:
            if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                (8, 2), (7, 3), (8, 4)]:
                return None, None, None, valid
    if abs(state[y][x]) in [2, 3, 4, 5, 8]:
        if (ny, nx) in [(3, 1), (3, 2), (3, 4), (3, 5),
                        (4, 1), (4, 2), (4, 4), (4, 5),
                        (5, 1), (5, 2), (5, 4), (5, 5)]:
            return None, None, None, valid
        if abs(state[ny][nx]) > abs(state[y][x]):
            if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                (8, 2), (7, 3), (8, 4)]:
                return None, None, None, valid
    if abs(state[y][x]) in [6, 7]:
        if (ny, nx) in [(3, 1), (3, 2), (3, 4), (3, 5),
                        (4, 1), (4, 2), (4, 4), (4, 5),
                        (5, 1), (5, 2), (5, 4), (5, 5)]:
            if direction == 0:
                ny -= 3
            if direction == 1:
                nx += 2
            if direction == 2:
                ny += 3
            if direction == 3:
                nx -= 2
        if abs(state[ny][nx]) > abs(state[y][x]):
            if (ny, nx) not in [(0, 2), (1, 3), (0, 4),
                                (8, 2), (7, 3), (8, 4)]:
                return None, None, None, valid
    if state[y][x] * state[ny][nx] > 0:
        return None, None, None, valid
    valid = True
    state[ny][nx] = state[y][x]
    state[y][x] = 0
    n_koma = [np.sum(state<0), np.sum(state>0)]
    reward, done = check_game_result(state, obs4, obs8, n_koma, turn)
    
    return state, reward, done, valid

def check_game_result(board, obs4, obs8, n_koma, turn):
    if board[0][3] > 0:
        return 1, True
    if board[8][3] < 0:
        return -1, True
    if n_koma[0] <= 0:
        return 1, True
    if n_koma[1] <= 0:
        return -1, True
    if turn >= 200:
        return 0, True
    if (board == obs8).all() and (board == obs4).all():
        return 0, True
    return 0, False

if __name__=="__main__":
    env = animalEnv()
    env.reset()
    print(env.board)
    while True:
        a = [int(i) for i in input().split()]
        reward, done, valid = env.step(a[0], a[1], a[2])
        
        if not valid:
            continue
        print(env.board)

        if done:
            break
    print(reward)