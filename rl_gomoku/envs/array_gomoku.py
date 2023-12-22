import numpy as np
import os
from typing import Tuple
from collections import deque


class ArrayGomoku(object):
    """board for the game"""
    _players = [1, 2]

    chess = {
        0: "_",
        1: "X",
        2: "O"
    }

    def __init__(self, board_size=15, length_win=5):
        self.width = board_size
        self.height = board_size
        self.n_in_row = length_win

        self.checkpoint: dict = ...
        self.states: dict = ...
        self.current_player = ...
        self.availables: list = ...
        self.last_move: int = ...
        self.last_moves: deque = ...
        self.reset()

    def reset(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = 1
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.last_moves = deque(maxlen=3)
        self.last_moves.extend([-1, -1, -1])
        self.checkpoint = None

    def save_checkpoint(self):
        self.checkpoint = {
            "states": self.states.copy(),
            "current_player": self.current_player,
            "availables": self.availables.copy(),
            "last_move": self.last_move,
            "last_moves": self.last_moves.copy()
        }

    def load_checkpoint(self):
        self.states = self.checkpoint["states"]
        self.current_player = self.checkpoint["current_player"]
        self.availables = self.checkpoint["availables"]
        self.last_move = self.checkpoint["last_move"]
        self.last_moves = self.checkpoint["last_moves"]

    def idx_to_location(self, idx):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = idx // self.width
        w = idx % self.width
        return h, w

    def location_to_idx(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move
    
    def current_board(self) -> np.ndarray:
        square_board = np.zeros((self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_black = moves[players == 1]
            move_white = moves[players == 2]
            square_board[move_black // self.width, move_black % self.height] = 1.    
            square_board[move_white // self.width, move_white % self.height] = 2.
        return square_board

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((7, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_black = moves[players == 1]
            move_white = moves[players == 2]
            
            square_state[0][move_black // self.width, move_black % self.height] = 1.    
            square_state[1][move_black[:-1] // self.width, move_black[:-1] % self.height] = 1.
            square_state[2][move_black[:-2] // self.width, move_black[:-2] % self.height] = 1.
            
            square_state[3][move_white // self.width, move_white % self.height] = 1.    
            square_state[4][move_white[:-1] // self.width, move_white[:-1] % self.height] = 1.
            square_state[5][move_white[:-2] // self.width, move_white[:-2] % self.height] = 1.

        if self.current_player == 1:
            # 1 stands for black.
            square_state[6][:, :] = 1.0
        return square_state[:, ::-1, :]

    def step(self, position_idx):
        # assert self.data[position_idx] == 0, "Invalid position"
        self.states[position_idx] = self.current_player
        self.availables.remove(position_idx)
        self.current_player = 3 - self.current_player
        self.last_move = position_idx

    def check_winner(self) -> Tuple[bool, int]:
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def terminated(self):
        """Check whether the game is ended or not"""
        win, winner = self.check_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def render(self, mode='human'):
        os.system("clear" if os.name == 'posix' else 'cls')  # Clear screen for both Unix/Linux and Windows
        square_state = np.zeros((self.width, self.height))
        for key, val in self.states.items():
            x, y = self.idx_to_location(key)
            square_state[x, y] = val
        # Adjust the first line to ensure correct alignment
        first_line = "   " + " ".join([f"{i:<2}" for i in range(self.width)])
        lines = [first_line]

        for i, row in enumerate(square_state):
            # Adjusting alignment for the row numbers and elements
            line_elements = []
            for j, cell in enumerate(row):
                if (i, j) == self.idx_to_location(self.last_move):
                    # Highlight the last move with a different color or symbol
                    # Example: ANSI color code for red text is '\033[91m'
                    line_elements.append(f"\033[91m{self.chess[cell]}\033[0m")
                else:
                    line_elements.append(self.chess[cell])
            line = f"{i:<2} " + '  '.join(line_elements)
            lines.append(line)

        last_line = f"last position: {self.last_move}, player: {self.current_player}"
        lines.append(last_line)
        output = "\n".join(lines)
        print(output)





