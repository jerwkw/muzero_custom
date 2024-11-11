import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        return self.board.copy()

    def get_legal_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, action):
        if self.done or self.board[action] != 0:
            return -1, self.done

        self.board[action] = self.current_player
        reward = self.check_winner()
        self.current_player = -self.current_player
        self.done = reward != 0 or not self.get_legal_actions()
        return reward, self.done

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return self.current_player
        if abs(sum(self.board.diagonal())) == 3 or abs(sum(np.fliplr(self.board).diagonal())) == 3:
            return self.current_player
        return 0  # No winner

    def get_state(self):
        return self.board.copy() * self.current_player
