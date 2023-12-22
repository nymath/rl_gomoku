import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Set
import copy


def position_to_idx(position, size=15):
    x, y = position
    return x * size + y


def idx_to_position(idx, size=15):
    return idx // size, idx % size


class GomokuEnv(gym.Env):
    chess = {
        0: "_",
        1: "X",
        2: "O"
    }

    def __init__(self, board_size=15, length_win=5):
        super().__init__()
        self.__last_move = None
        self._checkpoints = None
        self.length_win = length_win
        self.size = board_size
        self.board = np.zeros((self.size, self.size))
        # self.board = np.array([[0 for _ in range(self.size)] for _ in range(self.size)]) # 0 represents an empty cell
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.action_space = spaces.MultiDiscrete([self.size, self.size])  # Actions are positions on the board
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=int)  # Observations are board states
        self.winner = None
        self.last_move = (-1, -1)
        self.adjacent_size = 1
        self.black_list = []
        self.white_list = []
        self.moves = []
        self._adjacent_vacancies = set()
        self.episode_len = 0
        self.availables = list(range(self.size * self.size))

    def reset(
            self,
            *,
            seed: int | None = None,
            options=None,
    ):
        self.board = np.zeros((self.size, self.size))
        self.current_player = 1
        self.done = False
        self.winner = None
        self.last_move = (-1, -1)
        self.black_list = []
        self.white_list = []
        self.episode_len = 0
        self.availables = list(range(self.size * self.size))
        return np.array(self.last_move), {}

    def is_ended(self):
        return self.done

    def step(self, action):
        row, col = action
        position_idx = position_to_idx(action, self.size)

        if not self.is_valid_move(row, col):
            raise ValueError("Invalid move")

        self.board[row, col] = self.current_player
        self.episode_len += 1
        self.moves.append((row, col))
        self.last_move = (row, col)
        # Update adjacent vacancies after the move

        self._adjacent_vacancies.update(self.neighbor((row, col), self.adjacent_size))
        self.availables.remove(position_idx)
        reward = 0
        if self.check_winner(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1
        else:
            self.current_player = 3 - self.current_player  # Switch player
        return self.board, reward, self.done, False, {}

    def simulation_start(self, action):
        row, col = action
        self.board[row, col] = self.current_player
        self.__last_move = (row, col)

    def simulation_exit(self):
        row, col = self.__last_move
        self.board[row, col] = 0

    def render(self, mode='human'):
        os.system("clear" if os.name == 'posix' else 'cls')  # Clear screen for both Unix/Linux and Windows

        # Adjust the first line to ensure correct alignment
        first_line = "   " + " ".join([f"{i:<2}" for i in range(self.size)])
        lines = [first_line]

        for i, row in enumerate(self.board):
            # Adjusting alignment for the row numbers and elements
            line_elements = []
            for j, cell in enumerate(row):
                if (i, j) == self.last_move:
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
        logging.debug(output)
        print(output)

    def close(self):
        pass

    def is_valid_move(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == 0

    def check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal
        for dr, dc in directions:
            count = 1
            # Check in one direction
            for i in range(1, self.length_win):
                r, c = row + dr * i, col + dc * i
                if not (0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player):
                    break
                count += 1
            if count >= self.length_win:
                return True

            # Check in the opposite direction
            for i in range(1, self.length_win):
                r, c = row - dr * i, col - dc * i
                if not (0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == self.current_player):
                    break
                count += 1
                if count >= self.length_win:
                    return True
        return False

    def neighbor(self, position, radius):
        bias = list(range(-radius,  radius + 1))
        vacancies = set()
        for i in bias:
            if position[0] - i < 0 or position[0] - i >= self.size:
                continue
            for j in bias:
                if position[1] - j < 0 or position[1] - j >= self.size:
                    continue
                vacancies.add((position[0] - i, position[1] - j))
        return vacancies

    def adjacent_vacancies(self) -> Set[tuple]:
        # TODO: update this function
        # vacancies = set()
        # if self.moves:
        #     bias = list(range(-self.adjacent_size, self.adjacent_size + 1))
        #     for move in self.moves:
        #         for i in bias:
        #             if move[0] - i < 0 or move[0] - i >= self.size:
        #                 continue
        #             for j in bias:
        #                 if move[1] - j < 0 or move[1] - j >= self.size:
        #                     continue
        #                 vacancies.add((move[0] - i, move[1] - j))
        #     occupied = set(self.moves)
        #     vacancies -= occupied
        #
        my_vacancies = self._adjacent_vacancies - set(self.moves)
        # assert vacancies == my_vacancies
        return my_vacancies
        # return self._adjacent_vacancies

    def slow_save_checkpoints(self):
        self._checkpoints = {
            "board": self.board.copy(),  # 使用列表推导式创建浅副本
            "last_move": self.last_move,  # 元组不需要深度复制
            "moves": self.moves.copy(),  # 对于列表，使用copy()创建浅副本
            "current_player": self.current_player,
            "done": self.done,
            "winner": self.winner,
            "episode_len": self.episode_len,
            "availables": self.availables.copy(),
            "_adjacent_vacancies": self._adjacent_vacancies.copy()  # 如果是集合或列表，使用copy()
        }

    def slow_load_checkpoints(self):
        self.board = self._checkpoints["board"]
        self.last_move = self._checkpoints["last_move"]
        self.current_player = self._checkpoints["current_player"]
        self.done = self._checkpoints["done"]
        self.winner = self._checkpoints["winner"]
        self.moves = self._checkpoints["moves"]
        self.episode_len = self._checkpoints["episode_len"]
        self._adjacent_vacancies = self._checkpoints["_adjacent_vacancies"]
        self.availables = self._checkpoints["availables"]
